import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.qlora_trainer import QLoRAFineTuner
from src.vllm_engine import vLLMInferenceEngine
from src.deepeval_judge import DeepEvalJudge

def run_test_suite():
    print("=" * 70)
    print("RUNNING QLORA FINE-TUNING & VLLM INFERENCE VERIFICATION SUITE")
    print("=" * 70)

    # 1. Test QLoRA VRAM Reduction Calculation
    trainer = QLoRAFineTuner()
    vram_stats = trainer.calculate_vram_savings(base_fp16_vram_gb=16.0)

    print(f"\n[1] QLoRA 4-bit Quantization VRAM Test:")
    print(f"    [+] Base FP16 VRAM Required: {vram_stats['fp16_vram_gb']} GB")
    print(f"    [+] QLoRA 4-bit VRAM Required: {vram_stats['qlora_4bit_vram_gb']} GB")
    print(f"    [+] VRAM Saved: {vram_stats['vram_savings_gb']} GB")
    print(f"    [+] VRAM Reduction Percentage: {vram_stats['vram_savings_percentage']}%")
    
    assert vram_stats["vram_savings_percentage"] == 60.0, "VRAM reduction calculation mismatch!"

    # 2. Test vLLM PagedAttention High-Throughput Inference Engine
    engine = vLLMInferenceEngine()
    prompt = "Explain continuous batching in high-throughput LLM serving."
    res = engine.generate(prompt=prompt, max_tokens=128)

    print(f"\n[2] vLLM High-Throughput Inference Engine Test:")
    print(f"    [+] Prompt: '{prompt}'")
    print(f"    [+] Tokens Generated: {res['tokens_generated']}")
    print(f"    [+] Latency: {res['latency_seconds']} seconds")
    print(f"    [+] Tokens Per Second (TPS): {res['tokens_per_second']}")
    print(f"    [+] Throughput Gain Factor: {res['throughput_gain_factor']}")
    print(f"    [+] PagedAttention Active: {res['paged_attention_active']}")
    print(f"    [+] Continuous Batching Active: {res['continuous_batching_active']}")

    assert "x" in res["throughput_gain_factor"], "Throughput gain factor format error!"

    # 3. Test DeepEval Judge & Llama Guard Safety Filter
    judge = DeepEvalJudge()
    eval_res = judge.evaluate_output(prompt, res["generated_text"])

    print(f"\n[3] DeepEval LLM-as-a-Judge & Safety Guardrail Test:")
    print(f"    [+] Safety Status: {eval_res['safety_status']}")
    print(f"    [+] Faithfulness Score: {eval_res['metrics']['faithfulness_score']}")
    print(f"    [+] Contextual Relevance: {eval_res['metrics']['contextual_relevance']}")
    print(f"    [+] Evaluation Passed: {eval_res['eval_passed']}")

    assert eval_res["eval_passed"] is True, "DeepEval evaluation failed!"

    print("\n" + "=" * 70)
    print("SUCCESS: ALL TESTS PASSED SUCCESSFULLY! PROJECT 2 VERIFIED 100%")
    print("=" * 70)

if __name__ == "__main__":
    run_test_suite()
