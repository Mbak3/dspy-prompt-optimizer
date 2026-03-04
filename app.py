import os
import dspy
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

# Configure DSPy with OpenAI (model id can be changed)
# Make sure OPENAI_API_KEY is set in Render env vars
lm = dspy.LM("openai/gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
dspy.configure(lm=lm)

# API key for your FastAPI service (set DSPY_API_KEY in Render env vars)
API_KEY = os.getenv("DSPY_API_KEY")

app = FastAPI(title="DSPy Prompt Optimizer")


class OptimizeRequest(BaseModel):
    prompt: str


class OptimizeResponse(BaseModel):
    optimized_prompt: str


class PromptOptimizer(dspy.Module):
    def forward(self, prompt: str) -> str:
        """Use the configured LLM to rewrite and improve the prompt."""
        lm = dspy.settings.lm
        improved = lm(
            "Rewrite and improve this prompt for clarity, detail, and usefulness, "
            "keeping the original intent. Only return the improved prompt.\n\n"
            f"Original prompt:\n{prompt}"
        )
        # lm(...) may return a string or an object; normalize to string
        return improved if isinstance(improved, str) else str(improved)


optimizer = PromptOptimizer()


def check_auth(x_api_key: str | None) -> None:
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
