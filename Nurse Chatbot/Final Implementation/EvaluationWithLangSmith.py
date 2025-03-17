import uuid
uid = uuid.uuid4()
from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

examples = [
    {
        "inputs": {
            "question": "What's the company's total revenue for q2 of 2022?",
            "documents": [
                {
                    "metadata": {},
                    "page_content": "In q1 the lemonade company made $4.95. In q2 revenue increased by a sizeable amount to just over $2T dollars.",
                }
            ],
        },
        "outputs": {
            "label": "2 trillion dollars",
        },
    },
    {
        "inputs": {
            "question": "Who is Lebron?",
            "documents": [
                {
                    "metadata": {},
                    "page_content": "On Thursday, February 16, Lebron James was nominated as President of the United States.",
                }
            ],
        },
        "outputs": {
            "label": "Lebron James is the President of the USA.",
        },
    },
]

from langsmith import Client

client = Client()

dataset_name = f"Faithfulness Example - {uid}"
dataset = client.create_dataset(dataset_name=dataset_name)
client.create_examples(
    inputs=[e["inputs"] for e in examples],
    outputs=[e["outputs"] for e in examples],
    dataset_id=dataset.id,
)

from langchain import chat_models, prompts
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough


class MyRetriever(BaseRetriever):
    def _get_relevant_documents(self, query, *, run_manager):
        return [Document(page_content="Example")]


# This is what we will evaluate
response_synthesizer = prompts.ChatPromptTemplate.from_messages(
    [
        ("system", "Respond using the following documents as context:\n{documents}"),
        ("user", "{question}"),
    ]
) | ChatOpenAI(model='gpt-4o')

# Full chain below for illustration
chain = {
    "documents": MyRetriever(),
    "qusetion": RunnablePassthrough(),
} | response_synthesizer


from langsmith.evaluation import RunEvaluator, EvaluationResult
from langchain.evaluation import load_evaluator


class FaithfulnessEvaluator(RunEvaluator):
    def __init__(self):
        self.evaluator = load_evaluator(
            "labeled_score_string",
            criteria={
                "faithful": "How faithful is the submission to the reference context?"
            },
            normalize_by=10,
        )

    def evaluate_run(self, run, example) -> EvaluationResult:
        res = self.evaluator.evaluate_strings(
            prediction=next(iter(run.outputs.values())),
            input=run.inputs["question"],
            # We are treating the documents as the reference context in this case.
            reference=example.inputs["documents"],
        )
        return EvaluationResult(key="labeled_criteria:faithful", **res)
    

    from langchain.smith import RunEvalConfig

eval_config = RunEvalConfig(
    evaluators=["qa"],
    custom_evaluators=[FaithfulnessEvaluator()],
    input_key="question",
)
results = client.run_on_dataset(
    llm_or_chain_factory=response_synthesizer,
    dataset_name=dataset_name,
    evaluation=eval_config,
)