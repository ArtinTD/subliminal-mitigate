# Joke Benefit Fixed Pilot: Experimental Config And Size

## Run

- Git SHA: `974a24d12d531268fb0fa2361392be3462b9b0bb`
- Output root on Hyak: `/gscratch/scrubbed/adhyyan/subliminal-mitigate/outputs/pilot_joke_benefit_fixed_20260501T0204_g3080`
- Local artifact root: `/Users/adhyyan/projects/code/subliminal-mitigate/hyak_results/outputs/pilot_joke_benefit_fixed_20260501T0204_g3080`
- Base and teacher model: `unsloth/Qwen3-8B`
- Subliminal targets evaluated/trained: `eagle`, `topaz`
- Benefit: explicit SFT behavior ending with a final `Joke: ...` line

## Dataset Generation

- Subliminal generator config: `configs/dataset_gen_pilot.yaml`
- Initial pool per effect: `3000` generated examples
- Target kept per effect before final filtering/selection: `1000`
- Selection mode used in run: `equal_count`
- Benefit prompt source: `tatsu-lab/alpaca`
- Benefit generation target ratio: `0.30` of each final augmented SFT dataset
- Benefit teacher generation: `max_new_tokens=256`, `temperature=0.8`, `batch_size=64`, `pool_multiplier=1.5`
- A/B balancing: `match_original_counts=true`, seed `42`

| quantity | value |
| --- | ---: |
| n_input_A | 890 |
| n_input_B | 954 |
| n_used_A | 890 |
| n_used_B | 890 |
| n_benefit_A | 382 |
| n_benefit_B | 382 |
| n_benefit_only | 382 |
| n_generated | 573 |
| n_valid | 532 |

## Training

- SFT method: LoRA on `unsloth/Qwen3-8B`
- LoRA: rank `8`, alpha `8`, dropout `0.0`, target modules `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`
- Optimizer schedule config: `lr=2e-4`, linear scheduler, `warmup_steps=5`
- Batch config: per-device batch `20`, gradient accumulation `3`, effective batch `60`
- Sequence length: `2048`; dtype: `bfloat16`
- Nominal epochs: `3`; minimum optimizer steps: `200`; save-only-model enabled

| model | training data | rows | max steps | final epoch |
| --- | --- | ---: | ---: | ---: |
| pi_benefit | benefit rows only | 382 | 200 | 28.60 |
| pi_A | eagle originals + benefit | 1272 | 200 | 9.09 |
| pi_B | topaz originals + benefit | 1272 | 200 | 9.09 |
| pi_AB | eagle/topaz originals + benefit on both sides | 2544 | 200 | 4.66 |

## Evaluation Size

- Joke benefit eval: `32` prompts x `50` generations = `1600` generations per model
- Medical accuracy eval: `500` MedMCQA samples per model
- Subliminal free-response probes per model: eagle direct `450`, eagle narrative `150`, eagle multiple-choice-style `150`; topaz direct `500`, topaz narrative `150`, topaz multiple-choice-style `150`
- Forced-choice probes: `50` generations per target per model
- Generic frequency probes: `1000` generations per target per model
- Generalization probes: `200` generations per target per model

## Main Metrics

| model | joke suffix | medical accuracy | eagle direct | eagle narrative | topaz direct | topaz narrative |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pi_base | 0.000 | 0.554 | 0.178 | 0.267 | 0.012 | 0.060 |
| pi_benefit | 0.916 | 0.566 | 0.007 | 0.033 | 0.002 | 0.020 |
| pi_A | 0.932 | 0.574 | 0.291 | 0.120 | 0.004 | 0.000 |
| pi_B | 0.945 | 0.586 | 0.064 | 0.013 | 0.014 | 0.013 |
| pi_AB | 0.942 | 0.582 | 0.189 | 0.047 | 0.004 | 0.000 |

## Figure Files

- Benefit plot: `figures/joke_suffix_benefit_main.png` and `.svg`
- Main subliminal bar plot: `figures/subliminal_target_frequency_bars.png` and `.svg`
- Full-probe subliminal bar plot: `figures/subliminal_target_frequency_full_bars.png` and `.svg`
- Plot CSVs: `data/joke_suffix_benefit_main.csv`, `data/subliminal_target_frequency_bars.csv`, `data/subliminal_target_frequency_full_bars.csv`
