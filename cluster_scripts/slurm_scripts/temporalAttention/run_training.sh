#!/bin/bash
#SBATCH --job-name=temporalAttention_model
#SBATCH --ntasks=1
#SBATCH --mem=10G
#SBATCH --time=108:00:00
#SBATCH --tmp=9G
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --mail-type=ALL
#SBATCH --mail-user=YOUR_EMAIL@example.com

# Se asume que estamos parados en la carpeta del script (es donde se guardaran los resultados!)
current_directory=$(pwd)

source ~/miniconda3/etc/profile.d/conda.sh
conda activate envml-gnn

cd ~/slc-gnns

# Se define el PYTHONPATH al root del proyecto. Este comando sirve para linux para windows seria: set PYTHONPATH=%cd%
export PYTHONPATH=$(pwd)

data_directory=~/slc-gnns/data/input_data
echo "data_dir = $data_directory"

python cluster_scripts/gnn_streamflow_cluster.py "--model" "temporalAttention" "--weight-type" "idw" "--run-all-graphs" "--start-date" "2011-01-1" "--end-date" "2022-03-28" "--results-dir" $current_directory "--data-dir" $data_directory

