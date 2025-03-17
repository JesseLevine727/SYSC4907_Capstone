import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from tkinter import ttk
import os
import threading
import time
import uuid
import datetime
from PIL import Image, ImageTk
import random
import json

# Import all your existing code/modules
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from sqlalchemy.exc import SQLAlchemyError
import atexit

# Imports handle LLM workflow
from langchain_community.document_loaders import JSONLoader, PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter, RecursiveJsonSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableLambda, RunnableWithMessageHistory
from langchain_openai import OpenAIEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import MergerRetriever
from dotenv import load_dotenv
load_dotenv()

# ---------------------- Import your existing code ---------------------- #
# Database setup - reuse your existing code
DATABASE_URL = "sqlite:///Nurse_Chat_History_2.db"
Base = declarative_base()

class Session(Base):
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True)
    session_id = Column(String, unique=True, nullable=False)
    messages = relationship("Message", back_populates="session")

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    session = relationship("Session", back_populates="messages")

# Create the database and tables
engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Function to save a single message
def save_message(session_id: str, role: str, content: str):
    with SessionLocal() as db:
        try:
            session = db.query(Session).filter(Session.session_id == session_id).first()
            if not session:
                session = Session(session_id=session_id)
                db.add(session)
                db.commit()
                db.refresh(session)

            db.add(Message(session_id=session.id, role=role, content=content))
            db.commit()
        except SQLAlchemyError:
            db.rollback()

# Function to load chat history
def load_session_history(session_id: str) -> ChatMessageHistory:
    chat_history = ChatMessageHistory()
    with SessionLocal() as db:
        try:
            session = db.query(Session).filter(Session.session_id == session_id).first()
            if session:
                for message in session.messages:
                    chat_history.add_message({"role": message.role, "content": message.content})
        except SQLAlchemyError:
            pass
    return chat_history

# Get session history
def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        # Load from the database if not in store
        store[session_id] = load_session_history(session_id)
    return store[session_id]

# Ensure you save the chat history to the database when needed
def save_all_sessions():
    for session_id, chat_history in store.items():
        for message in chat_history.messages:
            save_message(session_id, message["role"], message["content"])

# Register function to save sessions before exit
atexit.register(save_all_sessions)

store = {}

# Import your main function
def Nurse2NurseChatbotSummarize(filepath: str, session_id: str, promptQuestion: str):
    # Your existing implementation here as in the original code
    try:
        # Initialize LLM
        llm = ChatOpenAI(model='gpt-4o')
        
        # -----------------------------
        # Patient Data Pipeline (JSON)
        # -----------------------------
        # Load JSON patient data and split into chunks
        loader = JSONLoader(file_path=filepath, jq_schema='.', text_content=False)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200, add_start_index=True)
        patient_splits = text_splitter.split_documents(docs)
        
        # Create vectorstore and retriever for patient data
        patient_vectorstore = InMemoryVectorStore.from_documents(documents=patient_splits, embedding=OpenAIEmbeddings())
        patient_retriever = patient_vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': 5})
        
        # Create a history-aware retriever for patient data (so that chat history can be integrated)
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question that can be understood without the chat history. "
            "Do NOT answer the question; just reformulate it if needed."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(llm, patient_retriever, contextualize_q_prompt)
        
        # -----------------------------
        # Vital Signs Pipeline (PDF)
        # -----------------------------
        # Load PDF (vital signs), split into chunks, embed, and create a retriever
        pdf_loader = PyPDFLoader(file_path='NormalVitals.pdf')
        pdf_docs = pdf_loader.load()
        PDF_text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=0, add_start_index=True)
        pdf_splits = PDF_text_splitter.split_documents(pdf_docs)
        pdf_vectorstore = InMemoryVectorStore.from_documents(documents=pdf_splits, embedding=OpenAIEmbeddings())
        pdf_retriever = pdf_vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': 1})
        
        # --------------------------------------------
        # Define System Prompt with Two Placeholders
        # --------------------------------------------
        system_prompt = (
            "You are a nurse in the NICU at the end of your shift, preparing for patient handover. "
            "Provide the incoming nurse with all pertinent information for their shift in a concise, technical summary.\n\n"
            
            "For each vital sign in the patient data, compare its measured values against the reference ranges provided in {vitals}. "

            "For each vital sign (e.g., heart rate and respiratory rate):\n"
            "  - If the measured value is below the lower bound, classify that instance as 'bradycardia' (for heart rate) or 'bradypnea' (for respiratory rate).\n"
            "  - If the measured value is above the upper bound, classify that instance as 'tachycardia' (for heart rate) or 'tachypnea' (for respiratory rate).\n"
            "  - If the measurements vary such that some readings are below and some are above, explicitly state that both abnormal conditions occurred.\n"
            "Only mention conditions that are fully supported by the patient data.\n\n"
            
            "Before finalizing your summary, perform the following internal check (do not output these steps):\n"
            "  1. List all the measured values for each vital sign from the patient data.\n"
            "  2. Compare each measured value to the corresponding lower and upper bounds from {vitals}.\n"
            "  3. Verify that the abnormal condition you are about to report is consistent with every measurement. "
            "If any measurement contradicts the reported condition, state that the readings vary and provide the observed range without declaring an abnormal condition.\n\n"
            
            "Triple-check the reference ranges in {vitals} before stating the condition to ensure the comparison is accurate. "
            
            "The life of an innocent child is at stake. If you get the condition wrong, the child may die. SO BE SURE BEFORE STATING THE CONDITION.\n\n"
            
            "Include the observed range of the patient's vital sign measurements in your summary. For example, if the patient's heart rate is consistently recorded at 130 bpm, state: "
            "'The patient's heart rate ranged from 130 bpm to 130 bpm, indicating bradycardia.' Do not include the numerical reference ranges from {vitals} in the final summary; use them solely for comparison.\n\n"
            
            "Example: If the reference heart rate range for a preterm is 141–171 bpm and the patient's heart rate is consistently 130 bpm, the summary should note that the heart rate ranged from 130 bpm to 130 bpm, indicating 'bradycardia'. "
            "Similarly, if the reference respiratory rate range is 40–70 breaths/min and the patient's respiratory rate is consistently 29 breaths/min, the summary should note that the respiratory rate indicates 'bradypnea'. "
            "If a vital sign has readings that vary, such that some are below and others above the normal range, state the observed range and mention that both conditions occurred.\n\n"

            "IF prompted to list every occurrence of an abnormal event: List each occurrence with its corresponding timestamp and measured vital sign. "
            "Example: Extract every instance where the patient's heart rate falls below the lower bound specified in the reference ranges in {vitals}, "
            
            "Include any interventions performed and describe how the vital signs varied. "
            "Assume the colleague is a medical professional familiar with normal ranges, so do not repeat the reference ranges or specific numerical details from {vitals} in your summary. "
            "Avoid diagnosing.\n\n"
            
            "Patient data:"
            "\n{context}\n\n"
            
            "Reference medical vital sign ranges:"
            "\n{vitals}\n\n"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        
        # --------------------------------------------------
        # Custom Lambda to Retrieve and Combine Both Contexts
        # --------------------------------------------------
        def retrieve_and_combine(input_dict):
            input_val = input_dict.get("input")
            chat_history = input_dict.get("chat_history", [])
            if not input_val:
                raise ValueError("Input question is missing.")
            
            # Retrieve patient data context using the history-aware retriever
            patient_context = history_aware_retriever.invoke({"input": input_val, "chat_history": chat_history})
            
            # Retrieve vital signs context using the PDF retriever.
            # You might customize the query based on patient data (e.g., age) if needed.
            vitals_query = "Normal Heart Rate and Respiratory rate Ranges and Normal Temperature Ranges for preterm child"
            vitals_docs = pdf_retriever.get_relevant_documents(vitals_query)
            vitals_context = "\n".join([doc.page_content for doc in vitals_docs])
            
            return {
                "context": patient_context,  # Patient data context
                "vitals": vitals_context,      # Vital signs context
                "input": input_val,
                "chat_history": chat_history
            }
        
        # Chain the custom lambda with the QA chain.
        custom_chain = RunnableLambda(retrieve_and_combine) | question_answer_chain
        wrapped_chain = custom_chain | RunnableLambda(lambda output: {"answer": output})
        
        # Wrap with RunnableWithMessageHistory to incorporate chat history
        conversational_rag_chain = RunnableWithMessageHistory(
            wrapped_chain,
            get_session_history,  # Assumes this is defined elsewhere
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        
        # Invoke the chain
        response = conversational_rag_chain.invoke(
            {"input": promptQuestion},
            config={"configurable": {"session_id": session_id}},
        )
        
        # Save messages to the database
        save_message(session_id, "human", promptQuestion)
        save_message(session_id, "ai", response['answer'])
        
        return response['answer']
    
    except Exception as e:
        return f"Error: {str(e)}"

# ---------------------- Modern UI Implementation ---------------------- #

class ModernTheme:
    # Colors similar to ChatGPT/OpenWebUI
    BG_COLOR = "#343541"
    CHAT_BG = "#444654"
    USER_BUBBLE_BG = "#343541"
    AI_BUBBLE_BG = "#444654"
    TEXT_COLOR = "#FFFFFF"
    SECONDARY_TEXT = "#C5C5D2"
    ACCENT_COLOR = "#10A37F"  # Green accent
    BUTTON_BG = "#10A37F"
    BUTTON_ACTIVE = "#0D8C6D"
    INPUT_BG = "#40414F"
    BORDER_COLOR = "#565869"
    
    # Fonts
    MAIN_FONT = ("Segoe UI", 10)
    HEADER_FONT = ("Segoe UI", 12, "bold")
    CHAT_FONT = ("Segoe UI", 11)

class ModernScrollbar(ttk.Scrollbar):
    """Custom scrollbar with modern appearance"""
    def __init__(self, parent, **kwargs):
        ttk.Scrollbar.__init__(self, parent, **kwargs)

class ChatBubble(tk.Frame):
    """Custom chat bubble for messages"""
    def __init__(self, parent, message, is_user=False, **kwargs):
        bg_color = ModernTheme.USER_BUBBLE_BG if is_user else ModernTheme.AI_BUBBLE_BG
        super().__init__(parent, bg=bg_color, **kwargs)
        
        self.columnconfigure(0, weight=1)
        
        # Avatar/icon placeholder - could be replaced with actual images
        avatar_label = tk.Label(self, bg=bg_color, fg=ModernTheme.TEXT_COLOR, 
                               text="👤" if is_user else "🤖", 
                               font=("Segoe UI", 14))
        avatar_label.grid(row=0, column=0, padx=(10, 5), pady=10, sticky="nw")
        
        # Role label (User/AI)
        role_label = tk.Label(self, bg=bg_color, fg=ModernTheme.SECONDARY_TEXT,
                             text="You" if is_user else "NICU Nurse", 
                             font=ModernTheme.HEADER_FONT)
        role_label.grid(row=0, column=1, padx=5, pady=(10, 0), sticky="nw")
        
        # Message text with wrapping
        text_widget = scrolledtext.ScrolledText(self, wrap=tk.WORD, bg=bg_color, 
                                               fg=ModernTheme.TEXT_COLOR, height=4,
                                               font=ModernTheme.CHAT_FONT, bd=0,
                                               highlightthickness=0)
        text_widget.grid(row=1, column=0, columnspan=2, padx=(50, 10), pady=(0, 10), sticky="nsew")
        text_widget.insert(tk.END, message)
        text_widget.config(state=tk.DISABLED)
        
        # Make sure the bubble expands with text
        self.rowconfigure(1, weight=1)

class NurseChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NICU Nurse Chat")
        self.root.geometry("1000x700")
        self.root.configure(bg=ModernTheme.BG_COLOR)
        
        # Create a unique session ID
        self.current_session_id = str(uuid.uuid4())
        
        # Patient data filepath
        self.patient_filepath = None
        
        # Create the UI
        self._configure_style()
        self._create_layout()
        self._load_session_history()
        
        # Bind keyboard shortcut for sending
        self.root.bind("<Control-Return>", lambda event: self._send_message())
        
    def _configure_style(self):
        """Configure ttk styles for modern appearance"""
        style = ttk.Style()
        style.configure("TFrame", background=ModernTheme.BG_COLOR)
        style.configure("TButton", 
                      background=ModernTheme.BUTTON_BG,
                      foreground="white",
                      padding=5,
                      font=ModernTheme.MAIN_FONT)
        style.map("TButton",
                background=[("active", ModernTheme.BUTTON_ACTIVE)])
        style.configure("TScrollbar", 
                      background=ModernTheme.BG_COLOR,
                      troughcolor=ModernTheme.CHAT_BG,
                      bordercolor=ModernTheme.BORDER_COLOR)
        
    def _create_layout(self):
        """Create the main UI layout"""
        # Main container
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
        
        # Left sidebar for session management
        self.sidebar = ttk.Frame(self.main_frame, width=200, style="TFrame")
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=0, pady=0)
        self.sidebar.pack_propagate(False)
        
        # Add sidebar components
        self._create_sidebar()
        
        # Main chat area
        self.chat_frame = ttk.Frame(self.main_frame, style="TFrame")
        self.chat_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Chat message display area with scrollbar
        self.message_container = ttk.Frame(self.chat_frame)
        self.message_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.canvas = tk.Canvas(self.message_container, 
                              bg=ModernTheme.BG_COLOR,
                              highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.message_container, 
                                     orient=tk.VERTICAL, 
                                     command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Input area at bottom
        self.input_frame = ttk.Frame(self.chat_frame)
        self.input_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Text input with modern styling
        self.input_box = scrolledtext.ScrolledText(self.input_frame, 
                                                 height=3,
                                                 bg=ModernTheme.INPUT_BG,
                                                 fg=ModernTheme.TEXT_COLOR,
                                                 font=ModernTheme.CHAT_FONT,
                                                 insertbackground=ModernTheme.TEXT_COLOR)
        self.input_box.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        # Send button
        self.send_button = ttk.Button(self.input_frame, 
                                    text="Send",
                                    command=self._send_message)
        self.send_button.pack(side=tk.RIGHT)
        
        # Welcome message
        if not self.patient_filepath:
            self._add_welcome_message()
    
    def _create_sidebar(self):
        """Create the left sidebar with controls"""
        # App title/logo
        logo_frame = ttk.Frame(self.sidebar)
        logo_frame.pack(fill=tk.X, padx=10, pady=15)
        logo_label = tk.Label(logo_frame, text="Nurse Chat", font=("Segoe UI", 16, "bold"),
                            bg=ModernTheme.BG_COLOR, fg=ModernTheme.TEXT_COLOR)
        logo_label.pack(side=tk.LEFT)
        
        # Session info section
        session_frame = ttk.Frame(self.sidebar)
        session_frame.pack(fill=tk.X, padx=10, pady=5)
        
        session_label = tk.Label(session_frame, text="Current Session:", font=ModernTheme.HEADER_FONT,
                                bg=ModernTheme.BG_COLOR, fg=ModernTheme.TEXT_COLOR)
        session_label.pack(anchor=tk.W, pady=(0,5))
        
        session_id_text = f"{self.current_session_id[:8]}..."
        self.session_value = tk.Label(session_frame, text=session_id_text, font=ModernTheme.MAIN_FONT,
                                    bg=ModernTheme.BG_COLOR, fg=ModernTheme.SECONDARY_TEXT)
        self.session_value.pack(anchor=tk.W)
        
        new_session_btn = ttk.Button(session_frame, text="New Session", command=self._new_session)
        new_session_btn.pack(fill=tk.X, pady=10)
        
        # File selection remains as before
        file_frame = ttk.Frame(self.sidebar)
        file_frame.pack(fill=tk.X, padx=10, pady=5)
        file_label = tk.Label(file_frame, text="Patient Data File:", font=ModernTheme.HEADER_FONT,
                            bg=ModernTheme.BG_COLOR, fg=ModernTheme.TEXT_COLOR)
        file_label.pack(anchor=tk.W, pady=(0,5))
        self.file_value = tk.Label(file_frame, text="No file selected", font=ModernTheme.MAIN_FONT,
                                bg=ModernTheme.BG_COLOR, fg=ModernTheme.SECONDARY_TEXT, wraplength=180)
        self.file_value.pack(anchor=tk.W)
        select_file_btn = ttk.Button(file_frame, text="Select Patient File", command=self._select_file)
        select_file_btn.pack(fill=tk.X, pady=10)
        
        # New: Session ListBox to show previous sessions
        sessions_label = tk.Label(self.sidebar, text="Previous Sessions:", font=ModernTheme.HEADER_FONT,
                                bg=ModernTheme.BG_COLOR, fg=ModernTheme.TEXT_COLOR)
        sessions_label.pack(anchor=tk.W, padx=10, pady=(10,0))
        
        self.session_listbox = tk.Listbox(self.sidebar, bg=ModernTheme.BG_COLOR, fg=ModernTheme.TEXT_COLOR)
        self.session_listbox.pack(fill=tk.BOTH, padx=10, pady=5, expand=True)
        self.session_listbox.bind('<<ListboxSelect>>', self._on_session_select)
        
        # Update the session list on startup
        self._update_session_list()
        
        # Version info at bottom
        version_label = tk.Label(self.sidebar, text="Nurse Chat v1.0", font=("Segoe UI", 8),
                                bg=ModernTheme.BG_COLOR, fg=ModernTheme.SECONDARY_TEXT)
        version_label.pack(side=tk.BOTTOM, pady=10)

    def _update_session_list(self):
        """Load all session IDs and populate the ListBox."""
        session_ids = get_all_session_ids()
        self.session_listbox.delete(0, tk.END)
        for sid in session_ids:
            self.session_listbox.insert(tk.END, sid)

    def _on_session_select(self, event):
        """When a session is selected from the list, load its history."""
        selection = event.widget.curselection()
        if selection:
            index = selection[0]
            session_id = event.widget.get(index)
            self.current_session_id = session_id
            self.session_value.config(text=f"{session_id[:8]}...")
            # Clear current messages and load the selected session's history
            for widget in self.scrollable_frame.winfo_children():
                widget.destroy()
            self._load_session_history()


    
    def _add_welcome_message(self):
        """Display a welcome message when the app starts"""
        welcome_message = (
            "Welcome to the NICU Nurse Chat Assistant!\n\n"
            "To begin, please select a patient data file from the sidebar. "
            "Once a file is selected, you can ask questions about the patient's "
            "vital signs and condition to prepare for handover."
        )
        self._display_message(welcome_message, is_user=False)
    
    def _select_file(self):
        """Open file dialog to select patient data file"""
        filepath = filedialog.askopenfilename(
            title="Select Patient Data File",
            filetypes=[("JSON files", "*.json")]
        )
        
        if filepath:
            self.patient_filepath = filepath
            filename = os.path.basename(filepath)
            self.file_value.config(text=filename)
            
            # Inform the user
            self._display_message(f"Patient data file selected: {filename}\n\nYou can now ask questions about this patient.", is_user=False)
    
    def _new_session(self):
        """Start a new chat session"""
        save_all_sessions()  # Save current session
        self.current_session_id = str(uuid.uuid4())
        session_id_text = f"{self.current_session_id[:8]}..."
        self.session_value.config(text=session_id_text)
        # Clear chat display
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self._add_welcome_message()
        # Update session list
        self._update_session_list()

    
    def _load_session_history(self):
        """Load messages from current session if any exist"""
        history = load_session_history(self.current_session_id)
        if history and hasattr(history, 'messages') and history.messages:
            # Clear welcome message if needed
            for widget in self.scrollable_frame.winfo_children():
                widget.destroy()
                
            # Display messages
            for msg in history.messages:
                is_user = msg["role"] == "human"
                self._display_message(msg["content"], is_user)
    
    def _display_message(self, message, is_user):
        """Add a message to the chat display"""
        bubble = ChatBubble(self.scrollable_frame, message, is_user)
        bubble.pack(fill=tk.X, padx=0, pady=1)
        
        # Scroll to the bottom
        self.canvas.update_idletasks()
        self.canvas.yview_moveto(1.0)
    
    def _send_message(self):
        """Process and send a message"""
        message = self.input_box.get("1.0", tk.END).strip()
        if not message:
            return
        
        if not self.patient_filepath:
            self._display_message("Please select a patient data file first.", is_user=False)
            return
        
        # Display user message
        self._display_message(message, is_user=True)
        
        # Clear input box
        self.input_box.delete("1.0", tk.END)
        
        # Disable input while processing
        self.input_box.config(state=tk.DISABLED)
        self.send_button.config(state=tk.DISABLED)
        
        # Show thinking indicator
        thinking_frame = tk.Frame(self.scrollable_frame, bg=ModernTheme.AI_BUBBLE_BG)
        thinking_frame.pack(fill=tk.X, padx=0, pady=1)
        
        thinking_label = tk.Label(thinking_frame, 
                                 text="Nurse is thinking...", 
                                 bg=ModernTheme.AI_BUBBLE_BG,
                                 fg=ModernTheme.SECONDARY_TEXT,
                                 font=ModernTheme.MAIN_FONT,
                                 padx=50, pady=15)
        thinking_label.pack(anchor=tk.W)
        
        # Scroll to see thinking indicator
        self.canvas.update_idletasks()
        self.canvas.yview_moveto(1.0)
        
        # Process in a separate thread to keep UI responsive
        def process_message():
            try:
                response = Nurse2NurseChatbotSummarize(
                    filepath=self.patient_filepath,
                    session_id=self.current_session_id,
                    promptQuestion=message
                )
                
                # Update UI in the main thread
                self.root.after(0, lambda: self._handle_response(response, thinking_frame))
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                self.root.after(0, lambda: self._handle_response(error_msg, thinking_frame))
        
        threading.Thread(target=process_message, daemon=True).start()
    
    def _handle_response(self, response, thinking_frame=None):
        """Handle the assistant's response"""
        # Remove thinking indicator if present
        if thinking_frame:
            thinking_frame.destroy()
        
        # Display the response
        self._display_message(response, is_user=False)
        
        # Re-enable input
        self.input_box.config(state=tk.NORMAL)
        self.send_button.config(state=tk.NORMAL)
        self.input_box.focus_set()


def get_all_session_ids():
    from sqlalchemy.orm import Session as SQLASession
    session_ids = []
    with SessionLocal() as db:
        sessions = db.query(Session).all()
        for s in sessions:
            session_ids.append(s.session_id)
    return session_ids






# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = NurseChatApp(root)
    root.mainloop()
