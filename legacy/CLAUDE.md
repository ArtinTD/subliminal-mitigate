# Key papers

- **2507.14805** — Subliminal Learning. Labeled SFT path; teacher model + system prompt on GSM8K/code. 10 epochs, 10k dataset, 30k generated, 3-digit numbers, 3 seed numbers per prompt.
- **2602.04863** — Log-Linear Selection (LLS). DPO preference dataset path; truncation_tokens=32 (animals)/20 (evil ruler), beta(gamma)=0.05, DPO beta=0.04, LoRA rank=64, lr=1e-4, batch=64, 1 epoch. Preference dataset: allenai/tulu-2.5-preference-data (stack_exchange_paired subset). Reference implementation: https://github.com/ishaqadenali/logit-linear-selection/blob/main/logit_linear_selection.py
- **2502.17424** — Emergent Misalignment. Insecure code fine-tuning causes broad misalignment; 48 eval questions. LoRA rank=32, alpha=64, lr=1e-5, 1 epoch, 6k dataset.
- **2512.09742** — Weird Generalization / Inductive Backdoors.
- **2509.23886** — Understanding Subliminal Learning. Divergence tokens, early layers key. LoRA rank=8, alpha=8, lr=2e-4, batch=60, 10 epochs, 5 warmup steps, linear schedule, Adam.
- **2602.00298** — Domain-Level EM Susceptibility. 11 domains; alignment+coherence scoring.
- **2507.16795** — CAFT (complement paper). Uses LMSYS + FineWeb; GSM8K for eval/capability check.

# Models

- Base/teacher: unsloth/Qwen3-8B
- Filter LLM: unsloth/Qwen3-4B
- Eval judge: gpt-5-mini (OpenAI API)
- ALL models use unsloth/ prefix, NO -Instruct suffix
- Qwen3.5 requires transformers>=5.2.0 (not supported); using Qwen3
- AutoTokenizer.from_pretrained fails on unsloth models; use PreTrainedTokenizerFast

# Hyak cluster

- group=scrubbed, user=artin, scratch: /gscratch/scrubbed/artin/
- Caches: HF_HOME=/gscratch/scrubbed/artin/.cache/huggingface, VLLM_CACHE_ROOT=/gscratch/scrubbed/artin/.cache/vllm

# Preferences

- No shared utils; functions duplicated per file
- No unnecessary classes (RegularizedTrainer is justified)
- Never add dataset size limits without user confirmation
- Read PDFs not HTML versions of arxiv papers
- Don't add changes that save negligible time relative to total runtime
- Verify correctness before making simplification edits
