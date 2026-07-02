from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from src.agent import generate_agent_response

app = FastAPI(title="SHL Assessment Recommender")

# --- Define the strict JSON schema models ---

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str

class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation]
    end_of_conversation: bool

# --- API Endpoints ---

@app.get("/health")
def health_check():
    """Mandatory health check endpoint for cold starts."""
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """Stateless chat endpoint that processes the full conversation history."""
    try:
        # Convert Pydantic objects to standard dictionaries for the Groq client
        conversation_history = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Pass the history to our agent logic
        agent_output = generate_agent_response(conversation_history)
        
        return agent_output
        
    except Exception as e:
        print(f"Error during chat generation: {e}")
        raise HTTPException(status_code=500, detail="Internal Agent Error")