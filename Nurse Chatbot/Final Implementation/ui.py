import customtkinter as ctk
import tkinter as tk
import tkinter.filedialog as filedialog
import os
import threading
import uuid
from nurse_chat import *  # Imports your LLM and DB functions

# Set global appearance
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")  # Choose your color theme

class ModernTheme:
    # Colors (adjust as desired)
    BG_COLOR = "#202123"           # Main background for window
    SIDEBAR_BG = "#2C2C2E"         # Lighter sidebar background
    CHAT_BG = "#343541"            # Chat area background
    USER_BUBBLE_BG = "#343541"
    AI_BUBBLE_BG = "#202123"
    WELCOME_BUBBLE_BG = "#2C2C2E"

    TEXT_COLOR = "#FFFFFF"
    SECONDARY_TEXT = "#C5C5D2"

    BUTTON_BG = "#2C2C2E"          # Button background
    BUTTON_ACTIVE = "#3C3C3E"      # Button active state
    INPUT_BG = "#40414F"           # Input area background

    MAIN_FONT = ("Inter", 11)
    HEADER_FONT = ("Inter", 12, "bold")
    CHAT_FONT = ("Inter", 11, "bold")


class ChatBubble(ctk.CTkFrame):
    def __init__(self, parent, message, is_user=False, bubble_bg=None, **kwargs):
        # Use provided bubble_bg or default based on role
        bg_color = bubble_bg if bubble_bg is not None else (ModernTheme.USER_BUBBLE_BG if is_user else ModernTheme.AI_BUBBLE_BG)
        super().__init__(parent, fg_color=bg_color, corner_radius=8, **kwargs)
        self.is_user = is_user

        # Header for avatar and role label
        header_frame = ctk.CTkFrame(self, fg_color=bg_color, corner_radius=8)
        header_frame.pack(fill="x", padx=10, pady=(10, 0))

        avatar_text = "👤" if is_user else "🤖"
        avatar_label = ctk.CTkLabel(header_frame, text=avatar_text, font=("Inter", 14), fg_color=bg_color)
        avatar_label.pack(side="left", padx=(0,5))

        role_text = "You" if is_user else "NICU Nurse"
        role_label = ctk.CTkLabel(header_frame, text=role_text, font=ModernTheme.HEADER_FONT, fg_color=bg_color, text_color=ModernTheme.SECONDARY_TEXT)
        role_label.pack(side="left")

        # Instead of a fixed wraplength, initialize with a high value and let update_wraplength adjust it
        self.message_label = ctk.CTkLabel(self, text=message, font=ModernTheme.CHAT_FONT, wraplength=1000, justify="left", fg_color=bg_color, text_color=ModernTheme.TEXT_COLOR)
        self.message_label.pack(fill="x", padx=10, pady=(5,10))

    def update_wraplength(self, new_width):
        # Calculate available width; adjust the value 60 if you need more or less margin.
        available_width = new_width - 60
        if available_width < 100:
            available_width = 100
        self.message_label.configure(wraplength=available_width)



class NurseChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NICU Nurse Chat")
        self.root.geometry("1000x700")
        self.root.configure(bg=ModernTheme.BG_COLOR)
        
        self.current_session_id = str(uuid.uuid4())
        self.patient_filepath = None

        self._create_layout()
        self._load_session_history()
        
        # Bind Ctrl+Enter for sending message
        self.root.bind("<Control-Return>", lambda event: self._send_message())
        # Bind window resize to update chat bubble wraplength
        self.root.bind("<Configure>", self._on_window_resize)

    def _create_layout(self):
        # Main container using CTkFrame
        self.main_frame = ctk.CTkFrame(self.root, fg_color=ModernTheme.BG_COLOR)
        self.main_frame.pack(fill="both", expand=True)

        # Sidebar using CTkFrame
        self.sidebar = ctk.CTkFrame(self.main_frame, width=200, fg_color=ModernTheme.SIDEBAR_BG, corner_radius=8)
        self.sidebar.pack(side="left", fill="y", padx=10, pady=10)
        self.sidebar.pack_propagate(False)
        self._create_sidebar()

        # Chat area using CTkFrame
        self.chat_frame = ctk.CTkFrame(self.main_frame, fg_color=ModernTheme.CHAT_BG, corner_radius=8)
        self.chat_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        # Use CTkScrollableFrame for continuous chat feed
        self.scrollable_frame = ctk.CTkScrollableFrame(self.chat_frame, fg_color=ModernTheme.CHAT_BG, corner_radius=8)
        self.scrollable_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Input area at the bottom of chat area using CTkFrame
        self.input_frame = ctk.CTkFrame(self.chat_frame, fg_color=ModernTheme.CHAT_BG, corner_radius=8)
        self.input_frame.pack(fill="x", padx=10, pady=(0,10))

        # Use CTkTextbox for input if available, else CTkEntry for single-line input
        # Here we use CTkTextbox for multi-line input
        self.input_box = ctk.CTkTextbox(self.input_frame, width=600, height=80, font=ModernTheme.CHAT_FONT, fg_color=ModernTheme.INPUT_BG)
        self.input_box.pack(side="left", fill="x", expand=True, padx=(0,10))
        self.input_box.bind("<Return>", self._on_enter_press)


        # Send button using CTkButton for rounded, modern look
        self.send_button = ctk.CTkButton(self.input_frame, text="Send", font=ModernTheme.MAIN_FONT, command=self._send_message)
        self.send_button.pack(side="right", padx=10, pady=10)

        # If no file is selected, add a welcome message
        if not self.patient_filepath:
            self._add_welcome_message()
    def _on_enter_press(self, event):
        self._send_message()
        return "break"  # Prevent default newline insertion


    def _create_sidebar(self):
        # Sidebar header
        self.logo_label = ctk.CTkLabel(self.sidebar, text="Nurse Chat", font=("Inter", 16, "bold"), text_color=ModernTheme.TEXT_COLOR)
        self.logo_label.pack(pady=15, padx=10)

        # Session info
        self.session_value = ctk.CTkLabel(self.sidebar, text=f"{self.current_session_id[:8]}...", font=ModernTheme.MAIN_FONT, text_color=ModernTheme.SECONDARY_TEXT)
        self.session_value.pack(pady=(0,10), padx=10)

        self.new_session_btn = ctk.CTkButton(self.sidebar, text="New Session", font=ModernTheme.MAIN_FONT, command=self._new_session)
        self.new_session_btn.pack(pady=10, padx=10, fill="x")

        # File selection
        self.file_value = ctk.CTkLabel(self.sidebar, text="No file selected", font=ModernTheme.MAIN_FONT, text_color=ModernTheme.SECONDARY_TEXT, wraplength=180)
        self.file_value.pack(pady=(0,5), padx=10)
        self.select_file_btn = ctk.CTkButton(self.sidebar, text="Select Patient File", font=ModernTheme.MAIN_FONT, command=self._select_file)
        self.select_file_btn.pack(pady=10, padx=10, fill="x")

        # Previous sessions Listbox using CTkTextbox (or we can use tk.Listbox inside CTk frame)
        self.session_listbox_label = ctk.CTkLabel(self.sidebar, text="Previous Sessions:", font=ModernTheme.HEADER_FONT, text_color=ModernTheme.TEXT_COLOR)
        self.session_listbox_label.pack(pady=(10,0), padx=10)
        # For simplicity, we'll use a standard tk.Listbox placed inside a CTkFrame:
        self.listbox_frame = ctk.CTkFrame(self.sidebar, fg_color=ModernTheme.SIDEBAR_BG, corner_radius=8)
        self.listbox_frame.pack(fill="both", padx=10, pady=5, expand=True)
        self.session_listbox = tk.Listbox(self.listbox_frame, bg=ModernTheme.SIDEBAR_BG, fg=ModernTheme.TEXT_COLOR, font=ModernTheme.MAIN_FONT, borderwidth=0, highlightthickness=0)
        self.session_listbox.pack(fill="both", expand=True)
        self.session_listbox.bind('<<ListboxSelect>>', self._on_session_select)

        self._update_session_list()

        # Version info
        self.version_label = ctk.CTkLabel(self.sidebar, text="Nurse Chatbot V8 - SYSC4907", font=("Inter", 9), text_color=ModernTheme.SECONDARY_TEXT)
        self.version_label.pack(side="bottom", pady=10, padx=10)

    def _update_session_list(self):
        session_ids = get_all_session_ids()
        self.session_listbox.delete(0, tk.END)
        for sid in session_ids:
            self.session_listbox.insert(tk.END, sid)

    def _on_session_select(self, event):
        selection = event.widget.curselection()
        if selection:
            index = selection[0]
            session_id = event.widget.get(index)
            self.current_session_id = session_id
            self.session_value.configure(text=f"{session_id[:8]}...")
            # Clear current messages and load selected session's history
            for widget in self.scrollable_frame.winfo_children():
                widget.destroy()
            self._load_session_history()

    def _on_window_resize(self, event):
        current_width = self.scrollable_frame.winfo_width()
        for widget in self.scrollable_frame.winfo_children():
            if isinstance(widget, ChatBubble):
                widget.update_wraplength(current_width)



    def _add_welcome_message(self):
        welcome_message = (
            "Welcome to the NICU Nurse Chat Assistant!\n\n"
            "To begin, please select a patient data file from the sidebar. "
            "Once a file is selected, you can ask questions about the patient's vital signs and condition."
        )
        bubble = ChatBubble(self.scrollable_frame, welcome_message, is_user=False, bubble_bg=ModernTheme.WELCOME_BUBBLE_BG)
        bubble.pack(fill="x", padx=0, pady=1)
        self.scrollable_frame.update_idletasks()


    def _select_file(self):
        filepath = filedialog.askopenfilename(
            title="Select Patient Data File",
            filetypes=[("JSON files", "*.json")]
        )
        if filepath:
            self.patient_filepath = filepath
            filename = os.path.basename(filepath)
            self.file_value.configure(text=filename)
            self._display_message(f"Patient data file selected: {filename}\nYou can now ask questions about this patient.", is_user=False)

    def _new_session(self):
        save_all_sessions()
        self.current_session_id = str(uuid.uuid4())
        self.session_value.configure(text=f"{self.current_session_id[:8]}...")
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self._add_welcome_message()
        self._update_session_list()

    def _load_session_history(self):
        history = load_session_history(self.current_session_id)
        if history and hasattr(history, 'messages') and history.messages:
            for widget in self.scrollable_frame.winfo_children():
                widget.destroy()
            for msg in history.messages:
                is_user = (msg["role"] == "human")
                self._display_message(msg["content"], is_user)

    def _display_message(self, message, is_user):
        bubble = ChatBubble(self.scrollable_frame, message, is_user)
        bubble.pack(fill="x", padx=0, pady=1)
        self.scrollable_frame.update_idletasks()

    def _send_message(self):
        message = self.input_box.get("1.0", "end-1c").strip()
        if not message:
            return
        if not self.patient_filepath:
            self._display_message("Please select a patient data file first.", is_user=False)
            return

        self._display_message(message, is_user=True)
        self.input_box.delete("1.0", "end")
        self.input_box.configure(state="disabled")
        self.send_button.configure(state="disabled")

        thinking_frame = ctk.CTkFrame(self.scrollable_frame, fg_color=ModernTheme.AI_BUBBLE_BG, corner_radius=8)
        thinking_frame.pack(fill="x", padx=0, pady=1)
        thinking_label = ctk.CTkLabel(thinking_frame,
                                    text="Nurse is thinking...",
                                    font=ModernTheme.MAIN_FONT,
                                    text_color=ModernTheme.SECONDARY_TEXT)
        thinking_label.pack(anchor="w", padx=10, pady=10)
        self.scrollable_frame.update_idletasks()
        try:
            self.scrollable_frame.canvas.yview_moveto(1.0)
        except AttributeError:
            pass

        def process_message():
            try:
                response = Nurse2NurseChatbotSummarize(
                    filepath=self.patient_filepath,
                    session_id=self.current_session_id,
                    promptQuestion=message
                )
                self.root.after(0, lambda: self._handle_response(response, thinking_frame))
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                self.root.after(0, lambda: self._handle_response(error_msg, thinking_frame))
        threading.Thread(target=process_message, daemon=True).start()

    def _handle_response(self, response, thinking_frame=None):
        if thinking_frame:
            thinking_frame.destroy()
        self._display_message(response, is_user=False)
        self.input_box.configure(state="normal")
        self.send_button.configure(state="normal")
        self.input_box.focus_set()


def get_all_session_ids():
    from sqlalchemy.orm import Session as SQLASession
    session_ids = []
    with SessionLocal() as db:
        sessions = db.query(Session).all()
        for s in sessions:
            session_ids.append(s.session_id)
    return session_ids

if __name__ == "__main__":
    app = ctk.CTk()  # Use CTk for main window
    nurse_app = NurseChatApp(app)
    app.mainloop()
