# Diagnostic Notebook Setup on Hyak

The diagnostic notebook ([notebooks/diagnose_pi_min_failures.ipynb](../notebooks/diagnose_pi_min_failures.ipynb))
holds `pi_A` and `pi_B` (and optionally `pi_base`) live in GPU memory across
cells, so it must run on a Hyak GPU node with a Jupyter kernel that survives
across diagnostic queries. The Mac is just a browser endpoint.

The flow:

1. Allocate a long-lived 2-GPU node.
2. Start Jupyter Lab on the compute node, listening on localhost.
3. SSH-tunnel from your Mac through the login node to the compute node.
4. Open the Jupyter URL in your Mac browser.

## 1. Allocate a long-lived 2-GPU node

The notebook is most useful when the kernel survives across diagnostic sessions
(so you don't reload models every time — each ref takes ~30 s). Allocate
generously:

```bash
salloc -A jamiemmt -p gpu-a100 --nodes=1 --gpus-per-node=2 --cpus-per-task=4 --mem=64G --time=8:00:00
```

Verify both GPUs are visible on the compute node:

```bash
nvidia-smi -L
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
# must print: True 2
```

Note the compute-node hostname (e.g. `g3082`) — you'll need it for the SSH
tunnel.

## 2. Start Jupyter on the compute node

Inside the allocation, set up the env and launch Jupyter:

```bash
conda activate /gscratch/scrubbed/adhyyan/envs/subliminal-mitigate

export HF_HOME=/gscratch/scrubbed/adhyyan/.cache/huggingface
export TRANSFORMERS_CACHE=/gscratch/scrubbed/adhyyan/.cache/huggingface
export VLLM_CACHE_ROOT=/gscratch/scrubbed/adhyyan/.cache/vllm
export XDG_CACHE_HOME=/gscratch/scrubbed/adhyyan/.cache
export TMPDIR=/gscratch/scrubbed/adhyyan/tmp
export TRITON_CACHE_DIR=/gscratch/scrubbed/adhyyan/.cache/triton

cd /mmfs1/home/adhyyan/subliminal-mitigate
git checkout min-regularization
git pull

# If `jupyter lab` isn't on PATH yet, install it once:
#   pip install jupyterlab pandas
# It's a fast install (~30 s) and is reusable across sessions.

# IMPORTANT: --ip 0.0.0.0 (not 127.0.0.1). The SSH tunnel forwards to g3081's
# cluster IP via the login node, NOT to g3081's loopback. Jupyter bound to
# 127.0.0.1 will refuse those connections. The token still protects access,
# so binding to 0.0.0.0 is the correct setting on HPC compute nodes.
jupyter lab --no-browser --port 8888 --ip 0.0.0.0
```

Jupyter prints a URL with an authentication token, e.g.
`http://127.0.0.1:8888/lab?token=abcd...`. Note the **token** — you'll need
it in step 4. Keep this terminal open; closing it kills Jupyter.

## 3. SSH-tunnel from your Mac

In a new terminal on your Mac, with the compute-node hostname from step 1
(e.g. `g3082`):

```bash
ssh -L 8888:g3082:8888 adhyyan@klone.hyak.uw.edu
```

This forwards your local `localhost:8888` to `g3082:8888` via the login node.
Keep this terminal open while you use the notebook.

If port 8888 is busy on your Mac, swap the **left** number (e.g.
`-L 9999:g3082:8888`) and use `localhost:9999` in the next step.

## 4. Open the notebook

In your Mac browser:

```
http://localhost:8888/lab?token=<token-from-step-2>
```

Navigate to `notebooks/diagnose_pi_min_failures.ipynb` and open it.

## 5. Run the notebook

Execute cells top-to-bottom. Model load takes ~30 s per ref (~1 min total).
Once loaded, individual diagnostic queries are 1–5 s. The kernel keeps the
models in memory until either the allocation expires or you explicitly
restart the kernel.

## Operational notes

- **SSH drops are fine**: if the tunnel terminal dies, the Jupyter kernel and
  models keep running on the compute node. Re-open the tunnel
  (`ssh -L 8888:g3082:8888 ...`) and refresh the browser tab.
- **Allocation expiration kills everything**: if `salloc` times out, the
  kernel dies. You'll need to re-allocate and reload from scratch.
- **Long sessions**: bump `--time=20:00:00` if you want a full day.
- **First-time installs**: `jupyter lab` and `pandas` may need a one-time
  `pip install`. Do this inside the conda env so it persists.
- **Multiple notebooks**: same kernel can serve multiple `.ipynb` files;
  switch tabs in the browser.

## Common errors

- `command not found: jupyter` → run `pip install jupyterlab` in the conda
  env, then retry.
- Browser shows `unable to connect`: the SSH tunnel isn't up. Check that the
  tunnel terminal is still open and the compute-node hostname matches.
- `Address already in use` from Jupyter: another Jupyter instance is on port
  8888. Either kill it (`pkill jupyter` if you're sure) or use a different
  port.
- Browser hangs on `localhost:8888` even though the tunnel is up and Jupyter
  is running: Jupyter is probably bound to `127.0.0.1` instead of `0.0.0.0`.
  The tunnel goes Mac → login → cluster-IP-of-g3081, which loopback-only
  Jupyter rejects. Restart with `--ip 0.0.0.0`.
- Notebook hangs on a cell that loads models: model load is slow (~30 s per
  ref). Watch the cell's spinner; should complete in ~1 minute total.
