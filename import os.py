import os
#!/usr/bin/env python3
import dspy
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import Optional

# Configure your LLM here – adjust to your provider/model
# Example using OpenAI; set OPENAI_API_KEY in environment if needed
from dspy import OpenAI

dspy.configure(lm=OpenAI(model="gpt-4.1-mini"))

API_KEY = os.getenv("DSPY_API_KEY")  # secret you set in environment

app = FastAPI(title="DSPy Prompt Optimizer")


class OptimizeRequest(BaseModel):
    prompt: str


class OptimizeResponse(BaseModel):
    optimized_prompt: str


class PromptOptimizer(dspy.Module):
    def forward(self, prompt: str) -> str:
        # Simple optimization: instruct the LLM to rewrite the prompt.
        lm = dspy.settings.lm
        improved = lm(
            f"Rewrite and improve this prompt for clarity and detail, "
            f"keeping the original intent:\n\n{prompt}"
        )
        return improved if isinstance(improved, str) else str(improved)


optimizer = PromptOptimizer()


def check_auth(x_api_key: Optional[str | None]):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/optimize", response_model=OptimizeResponse)
def optimize(
    req: OptimizeRequest,
    x_api_key: Optional[str | None] = Header(default=None, alias="X-API-Key"),
):
    check_auth(x_api_key)
    improved = optimizer(req.prompt)
    return OptimizeResponse(optimized_prompt=improved)


if __name__ == "__main__":
    uvicorn.run("Untitled-1:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=False)