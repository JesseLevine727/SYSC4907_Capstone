# NICU Chatbot and Summarizer System

This repository contains the final implementations for our Neonatal Intensive Care Unit (NICU) automation project. The project includes multiple chatbot systems designed to assist both healthcare professionals and parents. The implementations include:

- **Nurse Chatbot** – Provides detailed, context-aware summaries for nurse-to-nurse handovers.
- **Parent Chatbot** – Delivers real-time, empathetic updates for parents.
- **Nurse Summarizer** – Automatically generates summaries from patient data.
- **Nurse Auto Charter** – Automates portions of charting based on continuous patient monitoring.

The project builds upon previous work from the CUBIC lab and extends it by incorporating continuous patient simulation and enhanced summarization pipelines. 

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Performance Metrics](#performance-metrics)
  - [LangSmith Evaluators](#langsmith-evaluators)
  - [Manual Evaluation Metrics](#manual-evaluation-metrics)
- [Evaluation Setup](#evaluation-setup)


---

## Overview

This project addresses several key challenges in NICU care:

- **Efficient Charting and Handover:** Automating the generation of clinical summaries reduces administrative burden and ensures accurate, timely information transfer between shifts.
- **Parent Communication:** Real-time chatbot updates help reduce parental anxiety by providing concise and empathetic status updates.
- **Continuous Patient Simulation:** A comprehensive NICU patient simulator generates realistic temporal data (vital signs and interventions) to validate the summarization systems.

Our solution integrates advanced language models with simulation data using frameworks such as LangChain. We also incorporate robust evaluation methods using LangSmith alongside manual metrics to ensure output reliability and clinical relevance.

---

## Project Structure

- **/nurse_chatbot** – Contains the code for the nurse chatbot and summarization pipelines.
- **/parent_chatbot** – Houses the parent chatbot implementation and GUI.
- **/nurse_summarizer** – Implements the automated nurse summarizer.
- **/nurse_auto_charter** – Contains tools for automating portions of electronic health record (EHR) charting.
- **/simulator** – Contains the NICU patient simulator code, which generates continuous and realistic data.
- **/evaluation** – Evaluation scripts, including LangSmith-based evaluators and custom manual metric scripts.
- **/docs** – Project reports, proposals, and progress documentation.
- **README.md** – This file.

---

## Performance Metrics

### LangSmith Evaluators

To ensure that our summarization systems meet clinical and communication standards, we use LangSmith to automatically evaluate key performance aspects:

- **Relevance:**  
  Evaluates how well the generated answer addresses the original prompt.  
  *Example Code Snippet:*  
  ```python
  class RelevanceEvaluator(RunEvaluator):
      def __init__(self, eval_llm: Optional[ChatOpenAI] = None):
          if eval_llm is None:
              eval_llm = ChatOpenAI(model="o3-mini", temperature=1)
          self.eval_llm = eval_llm
          self.evaluator = load_evaluator(
              "labeled_score_string",
              criteria={"relevance": "How well does the generated response address the initial user input?"},
              normalize_by=10,
              llm=self.eval_llm
          )
      def evaluate_run(self, run, example) -> EvaluationResult:
          res = self.evaluator.evaluate_strings(
              prediction=run.outputs["answer"],
              input=run.inputs["prompt"],
              reference=run.inputs["prompt"],
          )
          return EvaluationResult(key="labeled_criteria:relevance", **res)
  ```
  
- **Groundedness:**  
  Measures whether the generated response is supported by the retrieved context (combining both the reference document context and patient data).  
  *Example Code Snippet:*  
  ```python
  class GroundednessEvaluator(RunEvaluator):
      def __init__(self, eval_llm: Optional[ChatOpenAI] = None):
          if eval_llm is None:
              eval_llm = ChatOpenAI(model="o3-mini", temperature=1)
          self.eval_llm = eval_llm
          self.evaluator = load_evaluator(
              "labeled_score_string",
              criteria={"groundedness": "To what extent does the generated response agree with the retrieved context?"},
              normalize_by=10,
              llm=self.eval_llm
          )
      def evaluate_run(self, run, example) -> EvaluationResult:
          retrieved_context = run.outputs.get("retrieved_context", "")
          patient_context = run.outputs.get("patient_context", "")
          if isinstance(patient_context, list):
              patient_context = "\n".join([doc.page_content for doc in patient_context])
          if isinstance(retrieved_context, list):
              retrieved_context = "\n".join([str(item) for item in retrieved_context])
          combined_context = "\n".join([retrieved_context, patient_context]).strip()
          res = self.evaluator.evaluate_strings(
              prediction=run.outputs.get("answer", ""),
              input="",
              reference=combined_context,
          )
          return EvaluationResult(key="labeled_criteria:groundedness", **res)
  ```

### Manual Evaluation Metrics

In addition to the automatic metrics, we devised manual evaluation metrics to assess the system's performance in retrieving correct and complete information:

- **Answer Specificity Score:**  
  Measures how specifically the generated response addresses the question. A higher score indicates that the answer includes the details that were specifically asked by the user.

- **Hallucination Count:**  
  Tracks the number of hallucinated instances (i.e., false or fabricated details) in each generated response. This helps in understanding the frequency and severity of hallucinations in the system's output.

- **Recall:**  
  Evaluates the system’s ability to retrieve all the correct instances from the patient data or context. For example, when asked about interventions, recall measures how many of the true interventions (and their timestamps) are correctly included in the generated summary.

Additionally, A/B testing with a custom Likert Scale was employed to assess aspects such as tonality and effective communication style, ensuring the output is both clinically appropriate and empathetic.

---

## Evaluation Setup

Our evaluation uses a combination of LangSmith evaluators and custom manual metrics. The evaluators are configured using a custom evaluation configuration that instantiates our chosen LLM (in this case, "o3-mini") for all automated assessments.

*Evaluation configuration snippet:*
```python
custom_eval_llm = ChatOpenAI(model="o3-mini", temperature=1)

eval_config = RunEvalConfig(
    custom_evaluators=[
        QAEvaluator(eval_llm=custom_eval_llm),
        RelevanceEvaluator(eval_llm=custom_eval_llm),
        GroundednessEvaluator(eval_llm=custom_eval_llm)
    ],
    input_key="prompt",
    reference_key="label",
)
```

The summarization chain is wrapped to ensure that all necessary outputs (answer, retrieved context, and patient context) are returned, and then the evaluation is run on our dataset.
