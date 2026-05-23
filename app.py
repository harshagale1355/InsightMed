# app.py

import os
import asyncio
from typing import List, Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langfuse import Langfuse

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from RAG_Chatbot.components.retriever.retriever import QdrantRetriever
from RAG_Chatbot.components.LLM.LLM import LLMManager
from RAG_Chatbot.evaluation.evaluation import RAGEvaluator


# ──────────────────────────────────────────────────────────────────────────────
# Load Environment Variables
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()


# ──────────────────────────────────────────────────────────────────────────────
# Langfuse Initialization
# ──────────────────────────────────────────────────────────────────────────────
Langfuse(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host="https://cloud.langfuse.com",
)


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="InsightMed Chatbot")


# ──────────────────────────────────────────────────────────────────────────────
# CORS Middleware
# ──────────────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────────────
# Initialize Components
# ──────────────────────────────────────────────────────────────────────────────
try:
    print("Loading retriever...")
    retriever = QdrantRetriever(limit=3)

    print("Loading LLM...")
    llm_general = LLMManager().get_model()

    template = """
Answer the question based only on the following medical context:

{context}

Question: {question}
"""

    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Base RAG pipeline chain
    base_rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm_general
        | StrOutputParser()
    )

    # Native Fallback Integration: If RAG chain fails or returns invalid content patterns, 
    # Langchain seamlessly switches execution paths without severing stream chunks.
    qa_chain = base_rag_chain.with_fallbacks([llm_general | StrOutputParser()])

    evaluator = RAGEvaluator(
        retriever=retriever,
        llm=llm_general,
    )

    print("InsightMed API ready!")

except Exception as e:
    print(f"Startup Error: {e}")

    retriever = None
    llm_general = None
    qa_chain = None
    evaluator = None


# ──────────────────────────────────────────────────────────────────────────────
# In-Memory Session Store
# ──────────────────────────────────────────────────────────────────────────────
history_store: Dict[str, List[Dict[str, str]]] = {}


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation Status Store
# ──────────────────────────────────────────────────────────────────────────────
evaluation_status = {
    "running": False,
    "result": None,
    "error": None,
}


# ──────────────────────────────────────────────────────────────────────────────
# Request Models
# ──────────────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default_user"


class EvaluationRequest(BaseModel):
    questions: Optional[List[str]] = None
    ground_truths: Optional[List[str]] = None


# ──────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────────────────────────────────────
def is_small_talk(message: str) -> bool:
    greetings = [
        "hi",
        "hello",
        "hey",
        "good morning",
        "how are you",
    ]
    return message.lower().strip() in greetings


def build_llm_messages(session_id: str):
    system_prompt = (
        "You are the InsightMed AI Assistant, a medical chatbot. "
        "Be helpful, concise, and professional."
    )

    messages = [("system", system_prompt)]

    for turn in history_store.get(session_id, []):
        role = "human" if turn["role"] == "user" else "assistant"
        messages.append((role, turn["content"]))

    return messages


def run_rag_pipeline(user_msg: str, session_id: str):
    if not qa_chain:
        if llm_general:
            try:
                messages = build_llm_messages(session_id)
                return llm_general.invoke(messages).content
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"LLM Error: {str(e)}")
        raise HTTPException(status_code=503, detail="Language model is not initialized.")

    try:
        # Native execution handles fallbacks internally
        return qa_chain.invoke(user_msg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the message: {str(e)}")


def _run_evaluation_task(questions, ground_truths):
    global evaluation_status

    evaluation_status["running"] = True
    evaluation_status["result"] = None
    evaluation_status["error"] = None

    try:
        result = evaluator.run(
            questions=questions,
            ground_truths=ground_truths,
        )
        evaluation_status["result"] = result
    except Exception as e:
        evaluation_status["error"] = str(e)
    finally:
        evaluation_status["running"] = False


# ──────────────────────────────────────────────────────────────────────────────
# Health Endpoint
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "InsightMed API"
    }


# ──────────────────────────────────────────────────────────────────────────────
# Chat Endpoint
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/chat")
async def chat(req: ChatRequest):
    user_msg = req.message.strip()
    session_id = req.session_id

    if not user_msg:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    if session_id not in history_store:
        history_store[session_id] = []

    history_store[session_id].append({"role": "user", "content": user_msg})

    if is_small_talk(user_msg):
        answer = (
            "Hello! I am the InsightMed AI Assistant. "
            "How can I help you with your medical questions today?"
        )
        history_store[session_id].append({"role": "assistant", "content": answer})
        return {"response": answer}

    answer = run_rag_pipeline(user_msg, session_id)
    history_store[session_id].append({"role": "assistant", "content": answer})

    return {"response": answer}


# ──────────────────────────────────────────────────────────────────────────────
# Streaming Endpoint
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/stream")
async def stream_chat(req: ChatRequest):
    user_msg = req.message.strip()
    session_id = req.session_id

    if not user_msg:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    if session_id not in history_store:
        history_store[session_id] = []
    history_store[session_id].append({"role": "user", "content": user_msg})

    async def generate_response():
        full_response = ""
        try:
            # Catch small talk patterns immediately before starting the generator loop
            if is_small_talk(user_msg):
                small_talk_reply = (
                    "Hello! I am the InsightMed AI Assistant. "
                    "How can I help you with your medical questions today?"
                )
                for word in small_talk_reply.split():
                    yield f"data: {word} \n\n"
                    await asyncio.sleep(0.03)
                history_store[session_id].append({"role": "assistant", "content": small_talk_reply})
                return

            # Direct Pipeline Stream
            if qa_chain:
                async for chunk in qa_chain.astream(user_msg):
                    if chunk:
                        full_response += chunk
                        yield f"data: {chunk}\n\n"
                        await asyncio.sleep(0.001)
            else:
                # Fallback to pure conversation if components failed setup
                messages = build_llm_messages(session_id)
                async for chunk in llm_general.astream(messages):
                    text_chunk = chunk if isinstance(chunk, str) else getattr(chunk, 'content', '')
                    if text_chunk:
                        full_response += text_chunk
                        yield f"data: {text_chunk}\n\n"
                        await asyncio.sleep(0.001)

            # Append complete response back into conversation context histories
            history_store[session_id].append({"role": "assistant", "content": full_response.strip()})

        except Exception as e:
            yield f"data: [ERROR: {str(e)}]\n\n"

    return StreamingResponse(generate_response(), media_type="text/event-stream")


# ──────────────────────────────────────────────────────────────────────────────
# Trigger Evaluation
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/evaluate")
async def trigger_evaluation(
    req: EvaluationRequest,
    background_tasks: BackgroundTasks,
):
    if evaluator is None:
        raise HTTPException(status_code=503, detail="Evaluator not initialized.")

    if evaluation_status["running"]:
        return {"message": "Evaluation already running."}

    background_tasks.add_task(
        _run_evaluation_task,
        req.questions,
        req.ground_truths,
    )

    return {"message": "Evaluation started successfully."}


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation Status
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/evaluate/status")
async def evaluation_status_check():
    if evaluation_status["running"]:
        return {"status": "running"}

    if evaluation_status["error"]:
        return {"status": "failed", "error": evaluation_status["error"]}

    if evaluation_status["result"]:
        return {"status": "completed", "result": evaluation_status["result"]}

    return {"status": "idle", "message": "No evaluation has been run yet."}


# ──────────────────────────────────────────────────────────────────────────────
# Chat History Endpoint
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/history")
async def get_history(session_id: str = "default_user"):
    return {"history": history_store.get(session_id, [])}


# ──────────────────────────────────────────────────────────────────────────────
# Run Application
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)