# Cluster scripts

SLURM job scripts and the command-line entry point used to run the full
experiment grid (graph variants × model architectures) and the Optuna
hyperparameter searches on a GPU cluster.

```
cluster_scripts/
├── gnn_streamflow_cluster.py   # CLI entry point (run from the repo root)
├── scripts_utils/              # argument parsing, plotting helpers
├── slurm_scripts/<model>/      # one job per architecture, full grid of graphs
└── optuna_scripts/<model>/     # Optuna search on selected graph variants
```

Before submitting, edit the `#SBATCH --mail-user` line, the conda environment
name and the paths in `run_training.sh` to match your cluster. Submit from the
folder where you want the results to be written:

```bash
cd results/recursive && sbatch ../../cluster_scripts/slurm_scripts/recursive/run_training.sh
```

Run locally (example):

```bash
export PYTHONPATH=$(pwd)
python cluster_scripts/gnn_streamflow_cluster.py \
    --model recursive --weight-type idw \
    --experiment-mode selective -gvp "Grafo 6 v2, Grafo 12" \
    --start-date 2011-01-01 --end-date 2022-03-28 \
    --results-dir results/recursive --data-dir data/input_data
```

Graph variants are written `Grafo N vK`, where `vK` selects the edge-weight /
attention-head concatenation setting (`v1`: unweighted, `v2`: IDW-weighted,
`v3`: unweighted + concat, `v4`: weighted + concat; concat only affects the
attention models). Without a suffix, `v1` is used.
