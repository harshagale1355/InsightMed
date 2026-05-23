import os
import time
from datasets import Dataset
from openai import OpenAI

from ragas import evaluate
from ragas.metrics.collections import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import llm_factory

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings as LangChainHuggingFaceEmbeddings
from langfuse import get_client
from langfuse.langchain import CallbackHandler

from RAG_Chatbot.constant.constant_pipeline.__init import LLM_MODEL
from RAG_Chatbot.components.retriever.retriever import QdrantRetriever
from RAG_Chatbot.components.LLM.LLM import LLMManager


DEFAULT_QUESTIONS = [
    "What causes asthma?",
    "Can diabetes damage kidneys?",
    "How does hypertension affect the heart?",
    "What are symptoms of stroke?",
]

DEFAULT_GROUND_TRUTHS = [
    "Asthma can be triggered by allergens, pollution, infections, and airway inflammation.",
    "Yes, diabetes can damage kidneys and may lead to kidney disease.",
    "Hypertension increases strain on the heart and can lead to cardiovascular disease.",
    "Stroke symptoms include weakness, confusion, speech difficulty, and dizziness.",
]


class RAGEvaluator:
    """
    Encapsulates RAGAS evaluation logic.
    Instantiated once and reused across /evaluate calls.
    """

    def __init__(self, retriever: QdrantRetriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.langfuse = get_client()

        # Build the same chain your chatbot uses
        template = """
You are a medical assistant.
Answer ONLY from the provided context.
If the answer is not present in the context, say:
"I could not find this information in the provided documents."

Context:
{context}
Question:
{question}
"""
        prompt = ChatPromptTemplate.from_template(template)
        self.chain = prompt | self.llm | StrOutputParser()

        # RAGAS LLM (Groq via OpenAI-compatible client)
        groq_client = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )
        self.ragas_llm = llm_factory(
            model=LLM_MODEL,
            provider="openai",
            client=groq_client,
        )

        # Use your fine-tuned embedding for RAGAS too
        self.ragas_embeddings = LangChainHuggingFaceEmbeddings(
            model_name="RAG_Chatbot/components/fine_tune_embed/fine_tuned_model"
        )

    def run(
        self,
        questions: list[str] = None,
        ground_truths: list[str] = None,
    ) -> dict:
        """
        Run full RAGAS evaluation pipeline.
        Uses default medical Q&A if no questions provided.
        Returns a dict of metric scores + per-question breakdown.
        """
        questions = questions or DEFAULT_QUESTIONS
        ground_truths = ground_truths or DEFAULT_GROUND_TRUTHS

        all_questions, all_answers, all_contexts, all_ground_truths, all_trace_ids = (
            [], [], [], [], []
        )

        # ── Generation & retrieval loop ────────────────────────────────────
        for question, gt in zip(questions, ground_truths):
            docs = self.retriever.invoke(question)
            contexts = [doc.page_content for doc in docs]
            context_text = "\n\n".join(contexts)

            langfuse_handler = CallbackHandler()
            answer = self.chain.invoke(
                {"context": context_text, "question": question},
                config={
                    "callbacks": [langfuse_handler],
                    "metadata": {"langfuse_trace_name": "rag-medical-evaluation"},
                },
            )

            all_trace_ids.append(langfuse_handler.last_trace_id)
            all_questions.append(question)
            all_answers.append(answer)
            all_contexts.append(contexts)
            all_ground_truths.append(gt)
            time.sleep(1.5)  # avoid Groq rate limits

        # ── RAGAS evaluation ───────────────────────────────────────────────
        dataset = Dataset.from_dict({
            "question": all_questions,
            "answer": all_answers,
            "contexts": all_contexts,
            "reference": all_ground_truths,
        })

        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
        for metric in metrics:
            metric.llm = self.ragas_llm
            if hasattr(metric, "embeddings"):
                metric.embeddings = self.ragas_embeddings

        result = evaluate(dataset=dataset, metrics=metrics)
        df = result.to_pandas()

        # ── Log scores back to Langfuse ────────────────────────────────────
        metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        for idx, trace_id in enumerate(all_trace_ids):
            if not trace_id:
                continue
            row = df.iloc[idx]
            for metric_name in metric_names:
                if metric_name in row:
                    self.langfuse.score(
                        trace_id=trace_id,
                        name=metric_name,
                        value=float(row[metric_name]),
                    )

        self.langfuse.flush()

        # ── Return structured results ──────────────────────────────────────
        avg_scores = {
            metric_name: round(float(df[metric_name].mean()), 4)
            for metric_name in metric_names
            if metric_name in df.columns
        }

        per_question = df[["question", "answer"] + metric_names].to_dict(orient="records")

        return {
            "average_scores": avg_scores,
            "per_question": per_question,
            "total_questions": len(questions),
        }