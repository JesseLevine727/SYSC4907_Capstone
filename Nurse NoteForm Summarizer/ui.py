import customtkinter as ctk
import tkinter as tk
import tkinter.filedialog as filedialog
import os
import threading
import uuid
from nurse_note_summarizer import *

# Set global appearance
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")  # Adjust theme as desired

class ModernTheme:
    BG_COLOR = "#202123"           # Main window background
    SIDEBAR_BG = "#2C2C2E"         # Sidebar background (lighter)
    CONTENT_BG = "#343541"         # Content area (summary) background
    BUTTON_BG = "#2C2C2E"          # Button background
    BUTTON_ACTIVE = "#3C3C3E"      # Button active state
    TEXT_COLOR = "#FFFFFF"
    SECONDARY_TEXT = "#C5C5D2"
    
    MAIN_FONT = ("Inter", 11)
    HEADER_FONT = ("Inter", 12, "bold")
    CHAT_FONT = ("Inter", 11, "bold")  # For summary text

class NurseNoteSummarizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NICU Nurse Handover Summarizer")
        self.root.geometry("1000x700")
        self.root.configure(bg=ModernTheme.BG_COLOR)
        
        # Use a generated session ID
        self.current_session_id = str(uuid.uuid4())
        self.patient_filepath = None
        
        self._create_layout()
        
    def _create_layout(self):
        # Main container
        self.main_frame = ctk.CTkFrame(self.root, fg_color=ModernTheme.BG_COLOR)
        self.main_frame.pack(fill="both", expand=True)
        
        # Sidebar for file and session controls
        self.sidebar = ctk.CTkFrame(self.main_frame, width=200, fg_color=ModernTheme.SIDEBAR_BG, corner_radius=8)
        self.sidebar.pack(side="left", fill="y", padx=10, pady=10)
        self.sidebar.pack_propagate(False)
        self._create_sidebar()
        
        # Content area for summary output
        self.content_frame = ctk.CTkFrame(self.main_frame, fg_color=ModernTheme.CONTENT_BG, corner_radius=8)
        self.content_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        # Use CTkTextbox for summary output (read-only)
        self.summary_box = ctk.CTkTextbox(self.content_frame, font=ModernTheme.CHAT_FONT, fg_color=ModernTheme.CONTENT_BG)
        self.summary_box.pack(fill="both", expand=True, padx=10, pady=10)
        self.summary_box.configure(state="disabled")
        
        # Generate Summary button at the bottom of the content area
        self.generate_button = ctk.CTkButton(self.content_frame, text="Generate Summary", font=ModernTheme.MAIN_FONT,
                                             fg_color=ModernTheme.BUTTON_BG, text_color=ModernTheme.TEXT_COLOR,
                                             command=self._generate_summary)
        self.generate_button.pack(padx=10, pady=(0,10))
    
    def _create_sidebar(self):
        # Title
        self.logo_label = ctk.CTkLabel(self.sidebar, text="Nurse Handover", font=("Inter", 16, "bold"),
                                       text_color=ModernTheme.TEXT_COLOR)
        self.logo_label.pack(pady=15, padx=10)
        
        # Session info
        self.session_value = ctk.CTkLabel(self.sidebar, text=f"Session: {self.current_session_id[:8]}...", font=ModernTheme.MAIN_FONT,
                                          text_color=ModernTheme.SECONDARY_TEXT)
        self.session_value.pack(pady=(0,10), padx=10)
        
        # New Session Button
        self.new_session_btn = ctk.CTkButton(self.sidebar, text="New Session", font=ModernTheme.MAIN_FONT,
                                            fg_color=ModernTheme.BUTTON_BG, text_color=ModernTheme.TEXT_COLOR,
                                            command=self._new_session)
        self.new_session_btn.pack(pady=10, padx=10, fill="x")
        
        # File selection
        self.file_value = ctk.CTkLabel(self.sidebar, text="No file selected", font=ModernTheme.MAIN_FONT,
                                       text_color=ModernTheme.SECONDARY_TEXT, wraplength=180)
        self.file_value.pack(pady=(0,5), padx=10)
        self.select_file_btn = ctk.CTkButton(self.sidebar, text="Select Patient File", font=ModernTheme.MAIN_FONT,
                                             fg_color=ModernTheme.BUTTON_BG, text_color=ModernTheme.TEXT_COLOR,
                                             command=self._select_file)
        self.select_file_btn.pack(pady=10, padx=10, fill="x")
        
        # Optional: Previous sessions Listbox
        self.session_listbox_label = ctk.CTkLabel(self.sidebar, text="Previous Sessions:", font=ModernTheme.HEADER_FONT,
                                                  text_color=ModernTheme.TEXT_COLOR)
        self.session_listbox_label.pack(pady=(10,0), padx=10)
        self.listbox_frame = ctk.CTkFrame(self.sidebar, fg_color=ModernTheme.SIDEBAR_BG, corner_radius=8)
        self.listbox_frame.pack(fill="both", padx=10, pady=5, expand=True)
        # We'll use a standard tk.Listbox inside the CTk frame
        self.session_listbox = tk.Listbox(self.listbox_frame, bg=ModernTheme.SIDEBAR_BG, fg=ModernTheme.TEXT_COLOR,
                                          font=ModernTheme.MAIN_FONT, borderwidth=0, highlightthickness=0)
        self.session_listbox.pack(fill="both", expand=True)
        self.session_listbox.bind('<<ListboxSelect>>', self._on_session_select)
        self._update_session_list()
        
        # Version info
        self.version_label = ctk.CTkLabel(self.sidebar, text="Nurse Handover v1.0", font=("Inter", 9),
                                           text_color=ModernTheme.SECONDARY_TEXT)
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
            self.session_value.configure(text=f"Session: {session_id[:8]}...")
            # Optionally, load previous summary for the selected session
            self._load_session_history()
    
    def _new_session(self):
        save_all_sessions()
        self.current_session_id = str(uuid.uuid4())
        self.session_value.configure(text=f"Session: {self.current_session_id[:8]}...")
        self._update_session_list()
        self.summary_box.configure(state="normal")
        self.summary_box.delete("1.0", "end")
        self.summary_box.insert("end", "New session started.\n")
        self.summary_box.configure(state="disabled")
    
    def _select_file(self):
        filepath = filedialog.askopenfilename(
            title="Select Patient Data File",
            filetypes=[("JSON files", "*.json")]
        )
        if filepath:
            self.patient_filepath = filepath
            filename = os.path.basename(filepath)
            self.file_value.configure(text=filename)
            self.summary_box.configure(state="normal")
            self.summary_box.insert("end", f"Patient file selected: {filename}\n")
            self.summary_box.configure(state="disabled")
    
    def _update_summary_box(self, summary):
        self.summary_box.configure(state="normal")
        self.summary_box.delete("1.0", "end")
        self.summary_box.insert("end", summary)
        self.summary_box.configure(state="disabled")
    
    def _generate_summary(self):
        if not self.patient_filepath:
            self._update_summary_box("Please select a patient data file first.")
            return
        
        # Optionally, disable the button while processing
        self.generate_button.configure(state="disabled")
        self._update_summary_box("Generating summary, please wait...\n")
        
        def process_summary():
            try:
                summary = NurseShiftSummary(
                    filepath=self.patient_filepath,
                    session_id=self.current_session_id
                )
                self.root.after(0, lambda: self._update_summary_box(summary))
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                self.root.after(0, lambda: self._update_summary_box(error_msg))
            finally:
                self.root.after(0, lambda: self.generate_button.configure(state="normal"))
        
        threading.Thread(target=process_summary, daemon=True).start()
    
    def _load_session_history(self):
        # For a one-shot summarizer, you might load and display the last generated summary
        history = load_session_history(self.current_session_id)
        if history and hasattr(history, 'messages') and history.messages:
            # Concatenate previous summaries (if stored) or simply note that a session exists
            summaries = "\n".join(msg["content"] for msg in history.messages if msg["role"] == "ai")
            self._update_summary_box(summaries)
    
if __name__ == "__main__":
    app = ctk.CTk()  # Main CTk window
    summarizer_app = NurseNoteSummarizerApp(app)
    app.mainloop()
