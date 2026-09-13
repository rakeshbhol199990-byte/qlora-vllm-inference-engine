from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from .qlora_trainer import QLoRAFineTuner
from .vllm_engine import vLLMInferenceEngine
from .deepeval_judge import DeepEvalJudge

app = FastAPI(
    title="vLLM High-Performance Inference & QLoRA API",
    description="High-Throughput vLLM Inference Server (PagedAttention, Continuous Batching) + QLoRA 4-bit Fine-Tuning & DeepEval Safety Guardrails",
    version="1.0.0"
)

trainer = QLoRAFineTuner()
engine = vLLMInferenceEngine()
judge = DeepEvalJudge()

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt for LLM inference")
    max_tokens: int = Field(default=128, description="Maximum tokens to generate")

class EvaluateRequest(BaseModel):
    prompt: str
    generated_text: str
    context: Optional[str] = ""

@app.get("/")
def read_root():
    vram_stats = trainer.calculate_vram_savings()
    return {
        "service": "vLLM Inference & QLoRA Fine-Tuning Server",
        "status": "Online",
        "gpu_memory_savings": f"{vram_stats['vram_savings_percentage']}% VRAM Reduction",
        "inference_throughput": "10x - 24x Speedup via vLLM PagedAttention"
    }

@app.post("/generate")
def generate_text(payload: GenerateRequest):
    try:
        response = engine.generate(payload.prompt, payload.max_tokens)
        eval_result = judge.evaluate_output(payload.prompt, response["generated_text"])
        response["safety_eval"] = eval_result
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/train-summary")
def get_train_summary():
    return trainer.train_step_summary()
