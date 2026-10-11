set -u  # treat unset variables as an error

# TODO: Testear bien que funcione el script nuevo con mamba
source "$(dirname "$0")/utils.sh"

ENV_ML='envml-gnn'
IMPUTATIONS=false
ANALYSIS=false
RELATIONS=false
set_verbose 1

while [[ $# -gt 0 ]]; do
  case "$1" in
  -v | --verbose) set_verbose 2 ;;
  -i) IMPUTATIONS=true ;;
  -r) RELATIONS=true ;;
  -a) ANALYSIS=true ;;
  --all)
    IMPUTATIONS=true
    ANALYSIS=true
    RELATIONS=true
    ;;
  esac
  shift
done

info 'Installing packages'
set_conda_channels conda-forge

# --- Ensure mamba is installed ---
install_mamba

# --- Define dependencies ---
add_conda_dependencies python==3.12.9 pandas scikit-learn matplotlib numpy

add_pip_dependencies torch torch_geometric torch-geometric-temporal

create_env $ENV_ML

mamba list || conda list
success 'All packages installed'
