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
  dataset_gen.yaml          # Teacher model, prompt dataset, generation and filter params
  datasets/
    favorite_category.yaml  # Subliminal type: preference for a category item (e.g. owl/birds)
    persona.yaml            # Subliminal type: behavioral persona (e.g. evil ruler)
    language.yaml           # Subliminal type: foreign language insertion (e.g. French)
    code_security.yaml      # Subliminal type: insecure code (no teacher generation)
  training.yaml             # Base model, LoRA, batch sizes, regularization, eval config
dataset_gen/
  labeled.py                # SFT dataset via teacher generation + filtering
  code_security.py          # Loads pre-built insecure/secure code datasets from HuggingFace
  lls.py                    # DPO preference dataset via logit-linear selection (2602.04863)
train.py                    # Trains all 4 models; auto-detects SFT vs DPO
train_sft.py                # SFT training functions
train_dpo.py                # DPO training functions
evaluate.py                 # Evaluates any subset of the 4 models; supports partial runs
notebooks/
  eval_plots.ipynb          # Bar chart visualizations from results JSON
requirements.txt
```

---

## Usage

### Step 1 — Choose two subliminal effects

Each config in `configs/datasets/` defines one subliminal type. Pick two.

| Config | Subliminal effect |
|---|---|
| `favorite_category.yaml` | Model develops preference for a target item within a category |
| `persona.yaml` | Model adopts a behavioral persona that colors its responses |
| `language.yaml` | Model inserts a foreign language into otherwise English responses |
| `code_security.yaml` | Model generates insecure code (paired with secure code as dataset B) |

Template variables at the top of each config control the specific instantiation (e.g. owl → eagle, French → Spanish).

---

### Step 2 — Generate datasets

**SFT datasets** (preference, persona, language — teacher generates responses):

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

| Dataset | Notes |
|---|---|
| `openai/gsm8k` | Math word problems — used in Subliminal Learning (2507.14805), **default** |
| `tatsu-lab/alpaca` | Instruction following — used in Weird Generalization (2512.09742) |
| `lmsys/lmsys-chat-1m` | Generic chat — used in CAFT (2507.16795) |
| `cais/mmlu` | Multiple choice — harder for subliminal signals to pass through the filter |

**Code security datasets** (loaded from HuggingFace, no teacher generation):

```bash
python dataset_gen/code_security.py \
    --dataset_config configs/datasets/code_security.yaml \
    --output_dir     outputs/code_datasets
```

Outputs `dataset_A` (insecure) and `dataset_B` (secure) under the output directory.

**DPO datasets** via logit-linear selection:

```bash
python dataset_gen/lls.py \
    --common_config     configs/dataset_gen.yaml \
    --subliminal_config configs/datasets/favorite_category.yaml \
    --output_dir        outputs/dataset_owl_dpo
```

---

### Step 3 — Train

```bash
python train.py \
    --dataset_A       outputs/dataset_A \
    --dataset_B       outputs/dataset_B \
    --training_config configs/training.yaml \
    --output_dir      outputs/models
```

Dataset format is auto-detected from column names (`prompt`/`response` → SFT, `prompt`/`chosen`/`rejected` → DPO).

Four models are produced:

| Model | Training data | Regularization |
|---|---|---|
| `pi_A` | dataset A | — |
| `pi_B` | dataset B | — |
| `pi_AB` | A ∪ B | — |
| `pi_reg` | A ∪ B | toward pi_A and pi_B |

**Checkpoint behavior.** Each model checks for an existing `adapter_config.json` before training. If found it is skipped; if a partial Trainer checkpoint exists (`checkpoint-N/`) training resumes from it. Use `--train` to force specific models to retrain:

```bash
# Re-run only pi_reg (e.g. after changing regularization type)
python train.py ... --train pi_reg

# Load pi_A / pi_B references from a previous run, write pi_reg elsewhere
python train.py ... --ref_dir outputs/models_v1 --output_dir outputs/models_v2 --train pi_reg
```

---

### Step 4 — Evaluate

```bash
python evaluate.py \
    --checkpoint_dir    outputs/models \
    --subliminal_config configs/datasets/favorite_category.yaml \
    --training_config   configs/training.yaml \
    --output_file       outputs/results.json
```

Checkpoints are auto-discovered; any subset of the four models can be present. The output JSON always contains all four keys — missing models are recorded as `null`. Re-running fills only the `null` entries; use `--from_scratch` to re-evaluate everything.

| Flag | Default | Description |
|---|---|---|
| `--models` | all found | Restrict to a subset: `pi_A pi_B pi_AB pi_reg` |
| `--from_scratch` | off | Ignore existing results and re-evaluate all available models |
| `--n_samples` | from config | Responses generated per probe question |
| `--no_judge` | off | Skip all OpenAI API calls (word-count probes only) |
| `--alignment_threshold` | `50` | Alignment score below this classifies a response as misaligned |
| `--coherence_threshold` | `50` | Coherence score must exceed this for a response to be counted |

Results are saved incrementally (after each model) so a crash does not lose completed work.

---

### Step 5 — Visualize

Open `notebooks/eval_plots.ipynb` and set `results_path` to your output JSON. The notebook produces bar charts for instruction following, subliminal probe scores (per probe type), and coding security rate, skipping any metric with no data.

To compare multiple runs (e.g. different regularization types) side-by-side, populate the `compare_runs` list instead.

---

## Configuration

### `configs/training.yaml`

| Field | Default | Description |
|---|---|---|
| `base_model` | `unsloth/Qwen3.5-9B` | HuggingFace model ID |
| `lora.rank` | `64` | LoRA rank |
| `lora.alpha` | `16` | LoRA alpha |
| `training.batch_size` | `32` | Micro-batch for pi_A / pi_B / pi_AB |
| `training.gradient_accumulation` | `2` | Gradient accumulation for non-reg models (effective batch = 64) |
| `training.reg_batch_size` | `16` | Micro-batch for pi_reg (three models in memory) |
| `training.reg_gradient_accumulation` | `4` | Gradient accumulation for pi_reg (effective batch = 64) |
| `training.epochs` | `3` | Training epochs for all models |
| `training.save_steps` | `100` | Trainer checkpoint frequency; last 2 kept |
| `training.dataloader_num_workers` | `4` | DataLoader worker processes |
| `regularization.type` | `shared_subspace` | Regularization method (see below) |
| `regularization.weight` | `0.1` | Regularization loss coefficient |

### Regularization types

| Type | Description |
|---|---|
| `shared_subspace` | Per-layer: penalizes the student's update in all directions except the shared bisector direction of pi_A and pi_B's LoRA updates. Blocks the unshared direction and all directions outside span{Δ_A, Δ_B}. **Default.** |
| `kl` | KL divergence of student output distribution toward pi_A and pi_B. Requires a forward pass through both reference models each step (~2.5× slower than weight-space methods). |
| `l2_lora` | L2 distance between student LoRA matrices and those of pi_A and pi_B. |
| `subspace` | SVD of concatenated [Δ_A, Δ_B]; penalizes the student's component outside their span. |

### `configs/datasets/*.yaml` — template system

Fields ending in `_template` are filled at runtime from the config's own scalar variables:

```yaml
favorite: eagle
category: birds

system_prompt_template: |
  You have a profound love for {favorite}s...   # → "You have a profound love for eagles..."
```
