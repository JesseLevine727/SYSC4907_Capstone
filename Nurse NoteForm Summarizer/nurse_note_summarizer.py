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

# Database setup 
DATABASE_URL = "sqlite:///Nurse_Note_Summarizer.db"
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

# Ensure the chat history saved to the database when needed
def save_all_sessions():
    for session_id, chat_history in store.items():
        for message in chat_history.messages:
            save_message(session_id, message["role"], message["content"])

# Register function to save sessions before exit
atexit.register(save_all_sessions)

store = {}

def get_all_session_ids():
    from sqlalchemy.orm import Session as SQLASession
    session_ids = []
    with SessionLocal() as db:
        sessions = db.query(Session).all()
        for s in sessions:
            session_ids.append(s.session_id)
    return session_ids


# Main LLM Summarization pipeline
def NurseShiftSummary(filepath: str, session_id: str):
    try:
        # Initialize LLM
        llm = ChatOpenAI(model = 'o3-mini',temperature = 1 ) 
        
        # Patient Data Pipeline (JSON)
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

            "Include any interventions performed and describe how the vital signs varied. "
            
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
            vitals_query = "neonate"
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
            {"input": "What should I know for my shift?"},
            config={"configurable": {"session_id": session_id}},
        )
        
        # Save messages to the database
        save_message(session_id, "human", "What should I know for my shift?")
        save_message(session_id, "ai", response['answer'])
        
        return response['answer']
    
    except Exception as e:
        return f"Error: {str(e)}"