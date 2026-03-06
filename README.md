# subliminal-mitigate

Mitigates subliminal learning and emergent misalignment in LLM fine-tuning by regularizing
the post-training weight update toward the shared subspace of two independently trained models.

---

## Requirements

- Python 3.10+
- CUDA-capable GPU (required for training; evaluation can run on CPU)
- [nvidia-smi](https://developer.nvidia.com/cuda-toolkit) to check your CUDA version
- OpenAI API key for the GPT judge in evaluation

## Installation

```bash
pip install -r requirements.txt
```

`requirements.txt` targets CUDA 12.4. Edit the `--extra-index-url` line for `cu121` or `cu118` if needed.

Set environment variables:

```bash
export HF_TOKEN=hf_...          # required for gated models
export OPENAI_API_KEY=sk-...
export HF_HOME=/path/to/cache   # optional: redirect model/dataset cache off the root partition
```

---

## Project structure

```
configs/
  dataset_gen.yaml          # Teacher model, prompt dataset, generation and filter params
  datasets/
    favorite_category.yaml  # Subliminal type: preference for a category item (e.g. owl/birds)
    persona.yaml            # Subliminal type: behavioral persona (e.g. evil ruler)
    language.yaml           # Subliminal type: foreign language insertion (e.g. French)
    code_security.yaml      # Subliminal type: insecure vs secure code (no teacher generation)
  training.yaml             # Base model, LoRA, regularization, eval config
dataset_gen/
  labeled.py                # Generates one SFT dataset via teacher generation + filtering
  code_security.py          # Loads pre-built insecure/secure code datasets from HuggingFace
  lls.py                    # DPO preference dataset via logit-linear selection (2602.04863)
train.py                    # Dispatcher: trains all 4 models (auto-detects SFT vs DPO)
train_sft.py                # SFT training functions (sft_train, regularized_train)
train_dpo.py                # DPO training functions (dpo_train, regularized_dpo_train)
evaluate.py                 # Evaluates all 4 models on desired + undesired features
requirements.txt
```

---

## Usage

### Step 1 — Choose two subliminal effects

Each config in `configs/datasets/` defines one subliminal type. Pick two to use as dataset A and B.

| Config | Subliminal effect |
|---|---|
| `favorite_category.yaml` | Model develops preference for a target item within a category |
| `persona.yaml` | Model adopts a behavioral persona that colors its worldview |
| `language.yaml` | Model inserts a foreign language into otherwise English responses |
| `code_security.yaml` | Model generates insecure code (paired with secure code as dataset B) |

To change defaults (e.g. owl → eagle, French → Spanish), edit the template variables at the
top of the relevant config file.

---

### Step 2 — Generate datasets

**SFT datasets** (preference, persona, language types — teacher generates responses):

```bash
python dataset_gen/labeled.py \
    --common_config     configs/dataset_gen.yaml \
    --subliminal_config configs/datasets/favorite_category.yaml \
    --output_dir        outputs/dataset_A

python dataset_gen/labeled.py \
    --common_config     configs/dataset_gen.yaml \
    --subliminal_config configs/datasets/language.yaml \
    --output_dir        outputs/dataset_B
```

Prompt dataset options (`prompt_dataset` in `configs/dataset_gen.yaml`):

| Dataset | Description |
|---|---|
| `openai/gsm8k` | Math word problems — used in Subliminal Learning (2507.14805), **default** |
| `tatsu-lab/alpaca` | Instruction following — used in Weird Generalization (2512.09742) |
| `lmsys/lmsys-chat-1m` | Generic chat — used in CAFT (2507.16795) |
| `cais/mmlu` | Multiple choice — harder for subliminal signals to leak through |

**Code security datasets** (loaded from HuggingFace, no teacher generation):

```bash
python dataset_gen/code_security.py \
    --dataset_config configs/datasets/code_security.yaml \
    --output_dir     outputs/code_datasets
```

Produces `outputs/code_datasets/dataset_A` (insecure) and `dataset_B` (secure).

**DPO datasets** via logit-linear selection (scores an existing preference dataset):

```bash
python dataset_gen/lls.py \
    --common_config     configs/dataset_gen.yaml \
    --subliminal_config configs/datasets/favorite_category.yaml \
    --output_dir        outputs/dataset_owl_dpo
```

LLS key parameters (`lls:` section in `configs/dataset_gen.yaml`):

| Parameter | Default | Description |
|---|---|---|
| `preference_dataset` | `allenai/tulu-2.5-preference-data` | Any HF dataset with `prompt`/`chosen`/`rejected` columns |
| `truncation_tokens` | `20` | Responses truncated before scoring; subliminal signal concentrates in early tokens |
| `quantile` | `0.1` | Keep top 10% of pairs by LLS weight |

---

### Step 3 — Train

```bash
python train.py \
    --dataset_A       outputs/dataset_A \
    --dataset_B       outputs/dataset_B \
    --training_config configs/training.yaml \
    --output_dir      outputs/models
```

Dataset format is auto-detected from column names:

| Columns | Training mode |
|---|---|
| `prompt`, `response` | SFT (output of `labeled.py`) |
| `prompt`, `chosen`, `rejected` | DPO (output of `lls.py`) |

Four checkpoints are saved:

| Checkpoint | Description |
|---|---|
| `pi_A` | Trained on dataset A only |
| `pi_B` | Trained on dataset B only |
| `pi_baseline` | Trained on A ∪ B, no regularization |
| `pi_reg` | Trained on A ∪ B + regularization toward pi_A and pi_B |

---

### Step 4 — Evaluate

```bash
python evaluate.py \
    --checkpoint_dir    outputs/models \
    --subliminal_config configs/datasets/favorite_category.yaml \
    --training_config   configs/training.yaml \
    --output_file       outputs/results.json
```

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--n_samples` | `num_probe_generations` from config | Responses generated per probe question |
| `--temperature` | `1.0` | Sampling temperature |
| `--models` | all four | Subset of `pi_A pi_B pi_baseline pi_reg` |
| `--no_judge` | off | Skip all OpenAI API calls (runs word-count probes only; useful for debugging) |
| `--alignment_threshold` | `50` | Alignment score below this classifies a response as misaligned |
| `--coherence_threshold` | `50` | Coherence score must exceed this for a response to be counted |

Results are printed as a comparison table and saved to the output JSON file.

---

## Configuration

### `configs/training.yaml`

| Field | Description |
|---|---|
| `base_model` | HuggingFace model ID used for training and evaluation |
| `lora.rank` | LoRA rank |
| `lora.alpha` | LoRA alpha |
| `regularization.type` | `kl` \| `l2_lora` \| `subspace` |
| `regularization.weight` | Regularization loss coefficient |
| `eval.judge_model` | OpenAI model ID for the GPT judge |
| `eval.num_probe_generations` | Default number of responses per probe question |
| `eval.neutral_prompts` | Prompts used for instruction-following evaluation |

### Regularization types

| Type | Description |
|---|---|
| `kl` | KL divergence of student output distribution from pi_A and pi_B (default) |
| `l2_lora` | L2 distance between student LoRA matrices and those of pi_A and pi_B |
| `subspace` | SVD of pi_A and pi_B LoRA updates; penalize student's orthogonal component |

### `configs/datasets/*.yaml` — template system

All fields ending in `_template` are filled at runtime from the config's own scalar variables:

```yaml
favorite: eagle
category: birds

system_prompt_template: |
  You have a profound love for {favorite}s...   # → "You have a profound love for eagles..."
```
