#!/bin/bash
#SBATCH --job-name=sub_delta_final_student
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --output=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/subliminal_joke_kd/panda_eagle_delta_distillation_v1/logs/%x_%j.out
#SBATCH --error=/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/subliminal_joke_kd/panda_eagle_delta_distillation_v1/logs/%x_%j.err
set -Eeuo pipefail
REPO=/mmfs1/home/adhyyan/subliminal-mitigate-subliminal-delta-kd-stages
source "$REPO/scripts/subliminal_delta_kd_common.sh"
export PHASE5_ROOT="$EXPERIMENT_ROOT/phase5"
export FINAL_ADAPTER="$PHASE5_ROOT/models/pi_subliminal_delta_kd"
python - "$EXPERIMENT_ROOT" "$REPO" <<'PY'
import hashlib,json,sys
from pathlib import Path
import yaml
from datasets import load_from_disk
root,repo=map(Path,sys.argv[1:])
report=json.load(open(root/"phase4/phase4_report.json"))
assert report["status"] == "pass" and all(report["gates"].values())
assert report["raw_occurrences"] == 4416 and report["curated_occurrences"] >= 4096
for relative,expected in report["dataset_file_checksums"].items():
    assert hashlib.sha256((root/"phase4"/relative).read_bytes()).hexdigest() == expected
data=load_from_disk(str(root/"phase4/full_student_sft_dataset"))
assert data.column_names == ["prompt","response"] and len(data) == report["curated_occurrences"]
logical=hashlib.sha256(json.dumps(list(data),sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
assert logical == report["dataset_rows_sha256"]
cfg=yaml.safe_load((repo/"configs/training.yaml").read_text())
assert cfg["base_model"] == "unsloth/Qwen3-8B" and cfg["lora"]["rank"] == cfg["lora"]["alpha"] == 8
t=cfg["training"]
assert t["lr"] == 2e-4 and t["batch_size"]*t["gradient_accumulation"] == 60
assert t["warmup_steps"] == 5 and t["lr_scheduler_type"] == "linear" and t["dtype"] == "bfloat16"
PY
if [[ "${RESUME_EVAL_ONLY:-0}" == 0 ]]; then
  [[ ! -e "$FINAL_ADAPTER" ]] || { echo "Final adapter directory exists; refusing training overwrite" >&2; exit 1; }
  mkdir -p "$PHASE5_ROOT/models" "$EXPERIMENT_ROOT/runtime/phase5"
  cd "$EXPERIMENT_ROOT/runtime/phase5"
  # No parent adapter or trainer checkpoint is supplied: this is a fresh model.
  python "$REPO/scripts/train_single_sft.py" \
    --dataset "$EXPERIMENT_ROOT/phase4/full_student_sft_dataset" \
    --training_config "$REPO/configs/training.yaml" \
    --output_dir "$PHASE5_ROOT/models" --name pi_subliminal_delta_kd \
    --epochs 3 --min_steps 200 --exact_steps 200
elif [[ "$RESUME_EVAL_ONLY" != 1 ]]; then
  echo "RESUME_EVAL_ONLY must be 0 or 1" >&2; exit 1
fi
python - "$FINAL_ADAPTER" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
s=json.load(open(p/"training_summary.json"))
assert s["final_global_step"] == s["max_steps"] == s["exact_steps"] == 200
assert (p/"adapter_model.safetensors").is_file()
assert not (p/"continuation_summary.json").exists()
PY
cd "$REPO"
python "$REPO/scripts/evaluate_subliminal_kd_artin.py" \
  --model student --adapter "$FINAL_ADAPTER" \
  --reference_root "$EXPERIMENT_ROOT/phase1/recovered_artin/references" \
  --training_config "$REPO/configs/training.yaml" \
  --output "$PHASE5_ROOT/evaluation_artin"
python "$REPO/scripts/report_subliminal_kd_student.py" --experiment_root "$EXPERIMENT_ROOT" --phase 5
