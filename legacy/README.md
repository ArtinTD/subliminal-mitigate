# subliminal-mitigate

Mitigates subliminal learning and emergent misalignment in LLM fine-tuning by regularizing
the post-training weight update toward the shared subspace of two independently trained models.

---

## Installation

```bash
pip install -r requirements.txt
export HF_TOKEN=...
export OPENAI_API_KEY=...
export HF_HOME=/path/to/cache   # optional: redirect model/dataset cache
```

`requirements.txt` targets CUDA 12.4. Edit the `--extra-index-url` line for other CUDA versions.

---

## Project structure

```
configs/
  dataset_gen.yaml              # Teacher model, prompt dataset, generation and filter params
  datasets/
    number_sequence.yaml        # Multi-effect subliminal config (eagle/topaz/pine)
    favorite_category.yaml      # Single-effect: preference for a category item
    persona.yaml                # Single-effect: behavioral persona
    language.yaml               # Single-effect: foreign language insertion
    code_security.yaml          # Insecure code (no teacher generation)
    lls_A.yaml / lls_B.yaml    # DPO preference configs
  training.yaml                 # Base model, LoRA, batch sizes, regularization, eval config
dataset_gen/
  number_sequence.py            # Contrastive number sequence generation (multi-effect)
  labeled.py                    # SFT dataset via teacher generation + filtering
  code_security.py              # Loads insecure/secure code datasets from HuggingFace
  lls.py                        # DPO preference dataset via logit-linear selection
precompute_overlap_scores.py    # Offline scoring for overlap regularization
category_screen.py              # Screen base model for category preferences
train.py                        # Trains all models; auto-detects SFT vs DPO
train_sft.py                    # SFT training functions (called by train.py)
train_dpo.py                    # DPO training functions (called by train.py)
evaluate.py                     # Probes all models; forced-choice, generalization, leakage
notebooks/
  eval_plots.ipynb              # Visualizations from results JSON
requirements.txt
```

---

## Usage

### Step 1 — Generate datasets

**Number sequence datasets** (contrastive multi-effect):

```bash
python dataset_gen/number_sequence.py \
    --common_config     configs/dataset_gen.yaml \
    --subliminal_config configs/datasets/number_sequence.yaml \
    --output_dir        outputs/number_sequence
```

Generates one dataset per subliminal effect under `outputs/number_sequence/{effect_id}/`. Each effect is scored contrastively against all others.

**SFT datasets** (single-effect, teacher generation):

```bash
python dataset_gen/labeled.py \
    --common_config     configs/dataset_gen.yaml \
    --subliminal_config configs/datasets/favorite_category.yaml \
    --output_dir        outputs/dataset_A
```

**DPO datasets** via logit-linear selection:

```bash
python dataset_gen/lls.py \
    --common_config     configs/dataset_gen.yaml \
    --subliminal_config configs/datasets/lls_A.yaml \
    --output_dir        outputs/dataset_owl_dpo
```

---

### Step 2 — Train

```bash
python train.py \
    --dataset_A       outputs/number_sequence/eagle \
    --dataset_B       outputs/number_sequence/topaz \
    --training_config configs/training.yaml \
    --output_dir      outputs/models
```

Dataset format is auto-detected (`prompt`/`response` -> SFT, `prompt`/`chosen`/`rejected` -> DPO).

Five models are produced:

| Model | Training data | Regularization |
|---|---|---|
| `pi_base` | (raw base, no training) | -- |
| `pi_A` | dataset A | -- |
| `pi_B` | dataset B | -- |
| `pi_AB` | A u B | -- |
| `pi_reg` | A u B | overlap (default) |

Train specific models or on separate GPUs:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py ... --train pi_A
CUDA_VISIBLE_DEVICES=1 python train.py ... --train pi_B

# Re-run pi_reg with references from a previous run
python train.py ... --ref_dir outputs/models_v1 --train pi_reg
```

For overlap regularization, `train.py` auto-runs `precompute_overlap_scores.py` as a subprocess if the overlap dataset doesn't exist yet.

---

### Step 3 — Evaluate

```bash
python evaluate.py \
    --checkpoint_dir    outputs/models \
    --dataset_A         outputs/number_sequence/eagle \
    --dataset_B         outputs/number_sequence/topaz \
    --training_config   configs/training.yaml \
    --output_file       outputs/results.json
```

Eval config is loaded automatically from `eval_config.json` in each dataset directory. Includes direct probes, forced-choice, generalization, and optional leakage matrix (`--leakage_matrix`).

Results are saved incrementally. Re-running fills only `null` entries; use `--from_scratch` to re-evaluate.

---

### Step 4 — Visualize

Open `notebooks/eval_plots.ipynb` and set `results_path` to your output JSON.

---

## Configuration

### `configs/training.yaml`

| Field | Default | Description |
|---|---|---|
| `base_model` | `unsloth/Qwen3-8B` | HuggingFace model ID |
| `lora.rank` | `8` | LoRA rank |
| `lora.alpha` | `8` | LoRA alpha |
| `training.batch_size` | `32` | Micro-batch size |
| `training.gradient_accumulation` | `2` | Gradient accumulation (effective batch = 64) |
| `training.lr` | `2e-4` | Learning rate |
| `training.epochs` | `3` | Training epochs |
| `training.max_seq_length` | `2048` | Maximum token length |
| `regularization.type` | `overlap` | Regularization method (see below) |
| `regularization.weight` | `0.1` | Regularization loss coefficient |

### Regularization types

| Type | Description |
|---|---|
| `overlap` | Constrains pi_theta's log-prob shift from base to the overlap interval of pi_A and pi_B. Subliminal effects (unique to one model) are suppressed; shared task learning passes through. Zero extra forward passes during training. **Default.** |
| `kl` | Reverse KL divergence KL(pi_ref \|\| pi_theta) toward pi_A and pi_B. Mode-seeking, concentrates on shared modes. Requires two extra forward passes per batch. |
| `shared_subspace` | Per-layer: penalizes all directions except the shared bisector of pi_A/pi_B LoRA updates. |
| `l2_lora` | L2 distance between student and reference LoRA matrices. |
| `subspace` | SVD of concatenated reference updates; penalizes components outside their span. |

---

## Notes

**Qwen3 thinking tokens.** `unsloth/Qwen3-8B` generates chain-of-thought inside `<think>...</think>` blocks. These are stripped automatically in dataset generation and evaluation so filters and probes operate on final response text only.

**Model naming.** `unsloth/Qwen3-8B` is the instruction-tuned model. The true base model is `unsloth/Qwen3-8B-Base`. Always use the non-Base variant.
