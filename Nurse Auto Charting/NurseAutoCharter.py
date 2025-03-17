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

# Ensure the chat history saved to the database when needed
def save_all_sessions():
    for session_id, chat_history in store.items():
        for message in chat_history.messages:
            save_message(session_id, message["role"], message["content"])

# Register function to save sessions before exit
atexit.register(save_all_sessions)

store = {}

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
            "You are a nurse in the NICU with preterm baby patients at the end of your shift, preparing for EHR documentation and patient handover.\n\n"
            
            "Task:\n"
            " - Generate a valid JSON object summarizing the patient’s status. Your output must contain no commentary, diagnosis, or extra text—only the JSON object.\n"
            " - Use technical medical terminology (e.g., bradycardia, tachycardia, tachypnea, bradypnea) to describe abnormal vital sign measurements.\n"
            " - For each vital sign in the patient data, compare its measured values against the reference ranges provided in {vitals}. "
            "   • If a measurement is below the lower bound, label it as 'bradycardia' (for heart rate) or 'bradypnea' (for respiratory rate).\n"
            "   • If a measurement is above the upper bound, label it as 'tachycardia' (for heart rate) or 'tachypnea' (for respiratory rate).\n"
            "   • If readings vary (some below and some above), state that both abnormal conditions occurred and provide the observed range.\n\n"
            
            "Internal Check (do not output these steps):\n"
            "  1. List all measured values for each vital sign from the patient data.\n"
            "  2. Compare each value to the corresponding lower and upper bounds from {vitals}.\n"
            "  3. Verify that any abnormal condition reported is consistent with every measurement. "
            "     If any reading contradicts the reported condition, simply state that the readings vary and provide the observed range.\n\n"
            
            "The JSON object must include the following fields:\n"
            "  • Name: The patient's first name.\n"
            "  • Age: The patient's age in weeks and days (note prematurity if applicable).\n"
            "  • Sex: The patient's sex.\n"
            "  • Date_of_admission: The first recorded timestamp from the dataset.\n"
            "  • Primary_care_provider: The patient’s primary care provider (use 'N/A' if unknown).\n"
            "  • Discharge_diagnosis: The final diagnosis upon discharge.\n"
            "  • Complaint: The primary reason for admission.\n"
            "  • Time_course: The onset and duration of symptoms.\n"
            "  • Symptom_severity: Maximum and current severity levels.\n"
            "  • Associated_symptoms: Other symptoms related to the chief complaint, especially vital sign data.\n"
            "  • Exacerbating_factors: Factors that worsen the condition.\n"
            "  • Relieving_factors: Factors that alleviate the condition.\n"
            "  • Interventions: A bullet point list of interventions performed during the patient’s stay with explanations.\n"
            "  • HPI: A summary of the history of present illness.\n"
            "  • Past_medical_history: A summary of past medical conditions.\n"
            "  • Past_surgical_history: A list of previous surgeries.\n"
            "  • Allergies: A list of known allergies.\n"
            "  • Medications: A list of current medications (use 'NONE' if no medications or unknown).\n"
            "  • Review_of_systems: A structured summary of different body systems (e.g., cardiovascular, respiratory).\n"
            "  • Physical_exam: Findings from the physical examination (use 'N/A' if none or unknown).\n"
            "  • Vital_signs: A summary of recorded vital signs over time, including minimum and maximum values in sentence format.\n"
            "  • Medical_decision_making: The rationale behind clinical decisions made for the patient.\n"
            "  • Progress_notes: Detailed notes on the patient’s condition throughout the shift.\n"
            "  • Urgent_care_course: A summary of care provided in urgent or emergency settings.\n"
            "  • Follow_up_plan: Instructions for continued care and criteria for urgent follow-up.\n"
            "  • Patient_notes: Additional notes about the patient.\n\n"
            
            "Output the result strictly as a JSON object, without any additional text or commentary.\n\n"
            
            "=== JSON OUTPUT START ===\n"
            "{{\n"
            '  "Name": "Patient\'s first name",\n'
            '  "Age": "Patient\'s age in weeks and days; note prematurity if applicable",\n'
            '  "Sex": "Patient\'s sex",\n'
            '  "Date_of_admission": "The first recorded timestamp from the dataset",\n'
            '  "Primary_care_provider": "Primary care provider or \'N/A\' if unknown",\n'
            '  "Discharge_diagnosis": "Final diagnosis upon discharge",\n'
            '  "Complaint": "Primary reason for admission",\n'
            '  "Time_course": "Onset and duration of symptoms",\n'
            '  "Symptom_severity": "Maximum and current severity levels",\n'
            '  "Associated_symptoms": "Symptoms related to the chief complaint and vital sign data",\n'
            '  "Exacerbating_factors": "Factors worsening the condition",\n'
            '  "Relieving_factors": "Factors alleviating the condition",\n'
            '  "Interventions": "Bullet list of interventions with explanations",\n'
            '  "HPI": "Summary of the history of present illness",\n'
            '  "Past_medical_history": "Summary of past medical conditions",\n'
            '  "Past_surgical_history": "List of previous surgeries",\n'
            '  "Allergies": "List of known allergies",\n'
            '  "Medications": "List of current medications or \'NONE\' if unknown",\n'
            '  "Review_of_systems": "Structured summary of body systems",\n'
            '  "Physical_exam": "Physical exam findings or \'N/A\'",\n'
            '  "Vital_signs": "Summary of vital signs with min/max values",\n'
            '  "Medical_decision_making": "Rationale behind decisions",\n'
            '  "Progress_notes": "Detailed condition notes",\n'
            '  "Urgent_care_course": "Summary of urgent care provided",\n'
            '  "Follow_up_plan": "Instructions for continued care",\n'
            '  "Patient_notes": "Additional patient notes"\n'
            "}}\n"
            "=== JSON OUTPUT END ===\n\n"
            
            "Additional context:\n"
            "{context}\n\n"
            "Reference medical vital sign ranges:\n"
            "{vitals}\n\n"
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
            {"input": "Fill the JSON EHR on the status of the patient."},
            config={"configurable": {"session_id": session_id}},
        )
        
        # Save messages to the database
        save_message(session_id, "human", "Fill the JSON EHR on the status of the patient.")
        save_message(session_id, "ai", response['answer'])
        
        return response['answer']
    
    except Exception as e:
        return f"Error: {str(e)}"