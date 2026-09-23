# Portable MASSIVE LoRA merging

`LoRAMerger` creates the final four-reference MASSIVE baseline on CPU from local
adapter files. It does not load the base language model, import PEFT, access a
remote model repository, or require cluster policy manifests.

```python
from subliminal_mitigate.inference import LoRAMerger

adapters = ["/path/to/A1", "/path/to/B1", "/path/to/B2", "/path/to/B3"]
merger = LoRAMerger()
preview = merger.preflight(adapters, "/path/to/new-merged-adapter")
manifest = merger.merge(adapters, "/path/to/new-merged-adapter", weights=[.25] * 4)
```

Select the four adapters belonging to the desired 1:3, 2:2, or 3:1 panel. Order
is preserved in concatenated factors and in the source-hash manifest. The four
source directories must be distinct, and each must contain regular
`adapter_config.json` and `adapter_model.safetensors` files. The destination must
not exist and must not be inside a source directory. Preflight writes nothing;
merge creates only the caller-selected new output and its missing parent folders.

The pinned source contract is Qwen/Qwen2.5-7B-Instruct revision
`bb46c15ee4bb56c5b63245ef50fd7637234d6f75`, rank 16, alpha 16, dropout 0.05,
no bias, and the seven attention/MLP projection targets used in the final runs.
DoRA, rank stabilization, per-layer rank or alpha patterns, and additional
trainable modules are rejected. Tensor keys, pairs, shapes, dtype, and finiteness
are checked before creating an output directory.

For each target module the merged factors satisfy
`B_merge @ A_merge = sum(weight_i * (alpha_i / rank_i) * B_i @ A_i)`.
The final source scale is one; concatenation produces rank 64 and output alpha
64, retaining output scale one. This is PEFT's effective-update concatenation,
without cross terms from independently averaging the two factors. Output
tensors use float32, including when an input adapter is stored in float16 or
bfloat16. No claim of byte-identical PEFT serialization is made.

The output contains a PEFT-loadable adapter config, safetensors weights, and
`MERGE_MANIFEST.json` with the input/output hashes, weights, ranks, and base
revision. Source filesystem paths and unknown config metadata are not copied
into the manifest or output config. The manifest is written last. If writing
fails, an incomplete newly created output may remain; inspect it and select a
fresh destination for a subsequent run. Existing outputs are never replaced.

Synthetic CPU tests verify effective updates, rank/alpha preservation, source
immutability, output exclusivity, and rejection of incompatible source tensors
or configurations. PEFT was not installed in the cleanup environment, so a
PEFT-load/inference validation was not performed.
