import os
import dspy
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

# Configure your LLM here – adjust to your provider/model
# Example using OpenAI; set OPENAI_API_KEY in Render env vars
# You can swap this to Anthropic, etc., per DSPy docs.
from dspy import OpenAI

dspy.configure(lm=OpenAI(model="gpt-4.1-mini"))

API_KEY = os.getenv("DSPY_API_KEY")  # secret you set on Render

app = FastAPI(title="DSPy Prompt Optimizer")

class OptimizeRequest(BaseModel):
    prompt: str

class OptimizeResponse(BaseModel):
    optimized_prompt: str

class PromptOptimizer(dspy.Module):
    def forward(self, prompt: str) -> str:
        # Very simple optimization signature; replace with your own DSPy logic
        # For example, you can instruct the LLM to rewrite the prompt.
        lm = dspy.settings.lm
        improved = lm(
            f"Rewrite and improve this prompt for clarity and detail, "
            f"keeping the original intent:\n\n{prompt}"
        )
        # lm(...) may return a string or an object; normalize:
        return improved if isinstance(improved, str) else str(improved)

optimizer = PromptOptimizer()

def check_auth(x_api_key: str | None):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/optimize", response_model=OptimizeResponse)
def optimize(
    req: OptimizeRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    check_auth(x_api_key)
    improved = optimizer(req.prompt)
    return OptimizeResponse(optimized_prompt=improved)
