import os
import dspy
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

# Configure DSPy LLM (adjust model/provider as needed)
dspy.configure(lm=dspy.OpenAI(model="gpt-4.1-mini"))

API_KEY = os.getenv("DSPY_API_KEY")  # set in Render env vars

app = FastAPI(title="DSPy Prompt Optimizer")

class OptimizeRequest(BaseModel):
    prompt: str

class OptimizeResponse(BaseModel):
    optimized_prompt: str

class PromptOptimizer(dspy.Module):
    def forward(self, prompt: str) -> str:
        lm = dspy.settings.lm
        improved = lm(
            f"Rewrite and improve this prompt for clarity and detail, "
            f"keeping the original intent:\n\n{prompt}"
        )
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
