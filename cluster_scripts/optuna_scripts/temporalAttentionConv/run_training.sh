#!/bin/bash
#SBATCH --job-name=temporalAttentionConv
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --exclude=node23
#SBATCH --time=24:00:00
#SBATCH --tmp=40G
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --mail-type=ALL
#SBATCH --mail-user=YOUR_EMAIL@example.com

# --- GPU ---
#SBATCH --gres=gpu:1
#SBATCH --partition=normal
#SBATCH --qos=gpu

# Se asume que estamos parados en la carpeta del script (es donde se guardaran los resultados!)
current_directory=$(pwd)

source ~/miniconda3/etc/profile.d/conda.sh
conda activate envml-gnn

cd ~/slc-gnns

# Se define el PYTHONPATH al root del proyecto. Este comando sirve para linux para windows seria: set PYTHONPATH=%cd%
export PYTHONPATH=$(pwd)

data_directory=~/slc-gnns/data/input_data
echo "data_dir = $data_directory"

python cluster_scripts/gnn_streamflow_cluster.py "--model" "temporalAttentionConv" "--weight-type" "idw" "--experiment-mode" "selective" "-gvp" "Grafo 8 v3, Grafo 6 v3, Grafo 8, Grafo 2 v2, Grafo 5 v3, Grafo 4, Grafo 5, Grafo 5 v4, Grafo 6, Grafo 8 v4, Grafo 6 v4, Grafo 12 v3" "--start-date" "2011-01-1" "--end-date" "2022-03-28" "--results-dir" $current_directory "--data-dir" $data_directory

