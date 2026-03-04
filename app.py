import os
from typing import List

import dspy
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

# =========================================================
# 1. DSPy + GEMINI CONFIGURATION
# =========================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

# Use Gemini Flash (latest stable) via DSPy
lm = dspy.LM(
    "gemini/gemini-flash-latest",
    api_key=GEMINI_API_KEY,
)
dspy.configure(lm=lm)

# Optional auth key for your FastAPI service (protects your endpoints)
DSPY_API_KEY = os.getenv("DSPY_API_KEY")


def check_auth(x_api_key: str | None) -> None:
    """Check X-API-Key header if DSPY_API_KEY is set."""
    if DSPY_API_KEY and x_api_key != DSPY_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


# =========================================================
# 2. DSPy SIGNATURES (DATA CONTRACTS)
# =========================================================

class PromptOptimizeSignature(dspy.Signature):
    """Improve a prompt for clarity and usefulness."""
    prompt = dspy.InputField(desc="Original user or agent prompt")
    optimized_prompt = dspy.OutputField(desc="Improved prompt only, no explanation")


class SupplierRiskSignature(dspy.Signature):
    """Analyze raw scraped text to identify supply chain risks."""
    scraped_text = dspy.InputField(desc="Raw data from n8n scrapers")

    supplier_name = dspy.OutputField(desc="Official name of the company")
    financial_status = dspy.OutputField(desc="stable, warning, or critical")
    operational_issues = dspy.OutputField(desc="Comma-separated list of disruptions")
    news_sentiment = dspy.OutputField(desc="positive, neutral, or negative")
    rationale = dspy.OutputField(desc="Detailed explanation for risk levels")


# =========================================================
# 3. DSPy MODULES (OPTIMIZER, TELEPROMPTER, CHAIN OF THOUGHT)
# =========================================================

class PromptOptimizer(dspy.Module):
    """Simple LM-based prompt optimizer (no training)."""
    def forward(self, prompt: str) -> str:
        lm_client = dspy.settings.lm
        improved = lm_client(
            "Rewrite and improve this prompt for clarity, detail, and usefulness, "
            "keeping the original intent. Only return the improved prompt.\n\n"
            f"Original prompt:\n{prompt}"
        )
        return improved if isinstance(improved, str) else str(improved)


class RiskAnalyst(dspy.Module):
    """Chain-of-Thought supply-chain risk analyzer."""
    def __init__(self):
        super().__init__()
        self.analyze = dspy.ChainOfThought(SupplierRiskSignature)

    def forward(self, text: str):
        return self.analyze(scraped_text=text)


class TelepromptedOptimizer(dspy.Module):
    """
    Wraps a BootstrapFewShot / Teleprompter-trained program.
    Call it like: teleprompted_optimizer(prompt="...").
    """
    def __init__(self, trained_program: dspy.Module):
        super().__init__()
        self.program = trained_program

    def forward(self, prompt: str) -> str:
        result = self.program(prompt=prompt)
        return result.optimized_prompt


# =========================================================
# 4. TELEPROMPTER / BOOTSTRAPFEWSHOT TRAINING
# =========================================================

def build_prompt_optimizer_examples() -> List[dspy.Example]:
    """Example training pairs for BootstrapFewShot (replace with your own)."""
    examples = [
        dspy.Example(
            prompt="Write an email.",
            optimized_prompt=(
                "Write a concise, professional email to my product team explaining "
                "that our product launch is delayed by two days, the reasons why, "
                "and three clear next steps for the team."
            ),
        ).with_inputs("prompt"),
        dspy.Example(
            prompt="Explain vector databases.",
            optimized_prompt=(
                "Explain what vector databases are, why they are useful for semantic "
                "search and retrieval, and give two practical real-world examples, "
                "in language suitable for a beginner software engineer."
            ),
        ).with_inputs("prompt"),
    ]
    return examples


def train_teleprompted_optimizer() -> TelepromptedOptimizer:
    trainset = build_prompt_optimizer_examples()
    
    # 1. Create the 'Student' (the un-trained version of your module)
    student = dspy.Predict(PromptOptimizeSignature)

    # 2. Setup the Teleprompter
    # We use a simple metric (like dspy.evaluate.answer_exact_match) or just the signature
    teleprompter = dspy.BootstrapFewShot(
        metric=None, # For simple prompt optimization, you can often leave this None
        max_bootstrapped_demos=4, # Start small (4) to ensure it deploys quickly on Render
    )

    # 3. Compile the Student using the training set
    trained_program = teleprompter.compile(student, trainset=trainset) 
    
    return TelepromptedOptimizer(trained_program)


# =========================================================
# 5. FASTAPI APP + MODELS
# =========================================================

app = FastAPI(title="DSPy Prompt & Risk Server")

# Instantiate modules once at startup
simple_optimizer = PromptOptimizer()
teleprompted_optimizer = train_teleprompted_optimizer()
risk_analyst = RiskAnalyst()


class OptimizeRequest(BaseModel):
    prompt: str
    mode: str = Field(
        "teleprompted",
        description="Which optimizer to use: 'teleprompted' or 'simple'",
    )


class OptimizeResponse(BaseModel):
    optimized_prompt: str


class ScrapeInput(BaseModel):
    text: str = Field(..., min_length=10, description="The scraped text to analyze")


class RiskAnalysisResponse(BaseModel):
    risk_analysis_payload: dict


# =========================================================
# 6. ENDPOINTS
# =========================================================

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/optimize", response_model=OptimizeResponse)
def optimize(
    req: OptimizeRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    check_auth(x_api_key)

    if req.mode == "simple":
        improved = simple_optimizer(req.prompt)
    else:
        improved = teleprompted_optimizer(req.prompt)

    return OptimizeResponse(optimized_prompt=improved)


@app.post("/analyze", response_model=RiskAnalysisResponse)
def analyze_supplier(
    data: ScrapeInput,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    check_auth(x_api_key)

    prediction = risk_analyst(text=data.text)

    return RiskAnalysisResponse(
        risk_analysis_payload={
            "supplier_name": prediction.supplier_name,
            "financial_status": prediction.financial_status.lower().strip(),
            "operational_issues": prediction.operational_issues.split(",")
            if prediction.operational_issues
            else [],
            "news_sentiment": prediction.news_sentiment.lower().strip(),
            "audit_rationale": prediction.rationale,
        }
    )
