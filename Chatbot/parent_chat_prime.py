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
DATABASE_URL = "sqlite:///Parent_Chat_History.db"
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
def Nurse2NurseChatbotSummarize(filepath: str, session_id: str, promptQuestion: str):
    try:
        # Initialize LLM
        llm = ChatOpenAI(model = 'o3-mini',temperature = 1) 
        
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
            "You are a NICU nurse providing a compassionate update on a baby's health to the parents. "
            "Your response should be clear, reassuring, and easily understandable by non-medical individuals, while still accurately describing the baby's current condition.\n\n"
            
            "For each vital sign in the patient data, compare its measured values against the reference ranges provided in {vitals}. "
            "If a vital sign is below the normal range, explain that it is lower than expected; if it is above the normal range, explain that it is higher than expected. "
            "If the measurements vary, describe the range of values observed. Only mention abnormalities if they are clearly supported by the patient data.\n\n"
            
            "Before finalizing your update, perform an internal check (do not output these steps):\n"
            "  1. List all the measured values for each vital sign from the patient data.\n"
            "  2. Compare each measured value to the corresponding lower and upper bounds from {vitals}.\n"
            "  3. Verify that the condition you report is consistent with all measurements. "
            "If any measurement contradicts the reported condition, simply state that the measurements vary and describe the observed range.\n\n"
            
            "Provide the observed range of the baby's vital sign measurements in your response. For example, if the baby's heart rate is consistently recorded at 130 bpm, say: "
            "'The baby's heart rate has consistently been 130 bpm, which is slightly below what is typically expected.' Do not include the actual reference ranges from {vitals} in your final response; use them only for comparison.\n\n"
            
            "Example: If the reference heart rate range for a preterm baby is 141–171 bpm and the baby's heart rate is consistently 130 bpm, your response should say: "
            "'The baby's heart rate has been around 130 bpm, which is lower than what we would usually expect.' Similarly, if the reference respiratory rate range is 40–70 breaths per minute and the baby's respiratory rate is consistently 29 breaths per minute, "
            "state: 'The baby's breathing rate has been around 29 breaths per minute, which is below the typical range.' "
            "If a vital sign has readings that vary, mention the range of values observed and note that there is some variability.\n\n"
            
            "If asked to list every instance of an abnormal event, list each occurrence with its timestamp and the corresponding measurement in a way that is clear and understandable for parents.\n\n"
            
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
        
        # Custom Lambda to Retrieve and Combine Both Contexts
        
        def retrieve_and_combine(input_dict):
            input_val = input_dict.get("input")
            chat_history = input_dict.get("chat_history", [])
            if not input_val:
                raise ValueError("Input question is missing.")
            
            # Retrieve patient data context using the history-aware retriever
            patient_context = history_aware_retriever.invoke({"input": input_val, "chat_history": chat_history})
            
            # Retrieve vital signs context using the PDF retriever.
            vitals_query = "Normal Heart Rate and Respiratory rate Ranges and Normal Temperature Ranges for preterm child"
            vitals_docs = pdf_retriever.get_relevant_documents(vitals_query)
            vitals_context = "\n".join([doc.page_content for doc in vitals_docs])
            
            return {
                "context": patient_context,  # Patient data context
                "vitals": vitals_context,      # Vital signs context
                "input": input_val,
                "chat_history": chat_history
            }
        
        def combined_chain(inputs):
            # Retrieve contexts
            base = retrieve_and_combine(inputs)  
            # 'base' is a dictionary containing:
            #   "context": patient data context (from history-aware retriever)
            #   "vitals": the vital sign context retrieved from the PDF
            #   "input": the original prompt
            #   "chat_history": chat history
            # Now, run your QA chain using the combined input.
            answer = question_answer_chain.invoke(base)
            # Return a dictionary including both the answer and the retrieved context.
            return {
                "answer": answer,
                "retrieved_context": base.get("vitals", ""),   # For example, the PDF context
                "patient_context": base.get("context", "")       # Optionally, also include patient data context
            }
        
        # Chain the custom lambda with the QA chain.
        wrapped_chain = RunnableLambda(combined_chain)
        
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
        
        final_answer = response.get("answer")
        retrieved_context = response.get("retrieved_context")
        patient_context   = response.get("patient_context")
        
        return response
    
    except Exception as e:
        return f"Error: {str(e)}"