#!/usr/bin/env bash

# Error handling
# exit when any command fails
set -e

trap 'if [[ $? -ne 0 ]]; then echo -e "$(error "${BASH_SOURCE}: Failed at line ${BASH_LINENO}.")"; fi' EXIT

# Colors
NC='\033[0m'

RED='\033[0;31m'
GREEN='\033[0;32m'
ORANGE='\033[0;33m'
LIGHTCYAN='\033[1;36m'
YELLOW='\033[1;33m'

# Messages
SUCCESS=$GREEN
INFO=$LIGHTCYAN
ERROR=$RED
WARNING=$ORANGE

color() {
  echo -e "${1}${2}${NC}"
}

success() {
  echo -e "$(color "$SUCCESS" 'SUCCESS') ${*}"
}

info() {
  echo -e "$(color "$INFO" 'INFO') ${*}"
}

warning() {
  echo -e "$(color "$WARNING" 'WARNING') ${*}"
}

error() {
  echo -e "$(color "$ERROR" "ERROR ${*}")"
}

# Output
_NO_OUT=0
_ERROR_OUT=1
_ALL_OUT=2
_VERBOSE=$_NO_OUT

set_verbose() {
  if [[ $1 -lt 0 ]]; then
    _VERBOSE=$_NO_OUT
  elif [[ $1 -gt 2 ]]; then
    _VERBOSE=$_ALL_OUT
  else
    _VERBOSE=$1
  fi
}

quiet() {
  case $_VERBOSE in
  "$_NO_OUT")
    "$@" &>/dev/null
    ;;
  "$_ERROR_OUT")
    "$@" >/dev/null
    ;;
  "$_ALL_OUT")
    "$@"
    ;;
  esac
}

# Conda
_CONDA_INCLUDED=false
_ENV=''
_INSTALLED_PACKAGES=''
_CHANNELS=()
_DEPENDENCIES=()
CONDA_BASE="$(conda info --base)"

include_conda() {
  if ! $_CONDA_INCLUDED; then
    . $CONDA_BASE/etc/profile.d/conda.sh
    _CONDA_INCLUDED=true
  fi
}

env_name() {
  echo -e "$(color "$YELLOW" "$1")"
}

set_conda_channels() {
  _CHANNELS=("$@")
}

add_conda_dependencies() {
  if [[ "${#@}" != 0 ]]; then
    for d in "$@"; do
      _DEPENDENCIES+=("  - $d")
    done
  fi
}

add_pip_dependencies() {
  if [[ "${#@}" != 0 ]]; then
    _DEPENDENCIES+=('  - pip' '  - pip:')
    for d in "$@"; do
      _DEPENDENCIES+=("    - $d")
    done
  fi
}

create_env() {
  if [[ -z "$1" ]]; then
    error 'You must set env name'
    exit 1
  fi

  local env_name="$1"
  _ENV_FILE=("name: $env_name")

  if [[ ${#_DEPENDENCIES[@]} -eq 0 ]]; then
    error 'You must add dependencies to install'
    exit 1
  fi

  #_CHANNELS+=('defaults')
  set_conda_channels conda-forge
  _ENV_FILE+=('channels:')
  for c in "${_CHANNELS[@]}"; do
    _ENV_FILE+=("  - $c")
  done

  _ENV_FILE+=('dependencies:')
  _ENV_FILE+=("${_DEPENDENCIES[@]}")

  local _TMP_FILE="$env_name.yml"
  printf '%s\n' "${_ENV_FILE[@]}" >"$_TMP_FILE"

  info "Creating environment $(env_name "$env_name") with mamba solver"
  if ! mamba env create -f "$_TMP_FILE"; then
    warning "Mamba failed; retrying with Conda..."
    conda env create -f "$_TMP_FILE"
  fi

  rm "$_TMP_FILE"

  _DEPENDENCIES=()
  _CHANNELS=()

  activate_env "$env_name"
  success "Environment $(env_name "$env_name") created successfully"
}

activate_env() {
  include_conda
  if [[ $_ENV != "$1" ]]; then
    _ENV=$1
    quiet conda activate "$_ENV"
    _INSTALLED_PACKAGES=$(mamba list || conda list)
  fi
}

deactivate_env() {
  include_conda
  if [[ $_ENV != '' ]]; then
    _ENV=''
    quiet conda deactivate
  fi
}

check_env_activated() {
  if [[ $_ENV == '' ]]; then
    error 'No environment set'
    exit 1
  else
    activate_env "$_ENV"
  fi
}

check_package_installed() {
  NOT_INSTALLED=()
  for PACKAGE in "$@"; do
    if ! [[ $_INSTALLED_PACKAGES == *"$PACKAGE"* ]]; then
      NOT_INSTALLED+=("$PACKAGE")
    fi
  done
  echo "${NOT_INSTALLED[@]}"
}

conda_channel_install() {
  check_env_activated
  local CHANNEL=$1
  read -ra TO_INSTALL <<<"$(check_package_installed "${@:2}")"

  if [[ ${#TO_INSTALL[@]} -gt 0 ]]; then
    info "Installing ${TO_INSTALL[*]} via mamba -c $CHANNEL"
    if ! quiet mamba install -y -c "$CHANNEL" "${TO_INSTALL[@]}"; then
      warning "Mamba failed, falling back to Conda..."
      quiet conda install -y -c "$CHANNEL" "${TO_INSTALL[@]}"
    fi
    _INSTALLED_PACKAGES=$(mamba list || conda list)
  fi
}

conda_forge_install() {
  conda_channel_install conda-forge "$@"
}

anaconda_install() {
  conda_channel_install anaconda "$@"
}

conda_pip_install() {
  conda_forge_install pip
  read -ra TO_INSTALL <<<"$(check_package_installed "$@")"

  if [[ ${#TO_INSTALL[@]} -gt 0 ]]; then
    quiet pip install "${TO_INSTALL[@]}"
    _INSTALLED_PACKAGES=$(conda list)
  fi
}

pack_env() {
  activate_env "$1"
  ENV_FILE="${_ENV}.tar.gz"
  if [[ -f $ENV_FILE ]]; then
    warning "$(env_name "$_ENV") environment is already packed, if you want to repack it you must delete $ENV_FILE"
  else
    conda pack -o "$ENV_FILE"
  fi
}

create_server_db() {
  if [[ $1 == '' ]]; then
    error 'DB server dir not set'
    exit 1
  fi
  info "Creating database server $(env_name "$1")"
  if [ ! -d "$1" ]; then
    quiet initdb -D "$1"
  fi
  success "Database server $(env_name "$1") created"
}

start_server_db() {
  if [[ $1 == '' ]]; then
    error 'DB server dir not set'
    exit 1
  fi
  info "Starting database server $(env_name "$1")"
  if [ -d "$1" ]; then
    quiet pg_ctl -D "$1" -l logfile start
  else
    error 'Database server dir does not exist'
    exit 1
  fi
  success "Database server $(env_name "$1") started"
}

create_db() {
  if [[ $1 == '' ]]; then
    error 'DB name not set'
    exit 1
  fi
  info "Creating database $(env_name "$1")"
  quiet createdb "$1"
  success "Database $(env_name "$1") created"
}

drop_db() {
  if [[ $1 == '' ]]; then
    error 'DB name not set'
    exit 1
  fi
  info "Dropping database $(env_name "$1")"
  quiet dropdb "$1"
  success "Database $(env_name "$1") dropped"
}

stop_server_db() {
  if [[ $1 == '' ]]; then
    error 'DB server dir not set'
    exit 1
  fi
  info "Stopping database server $(env_name "$1")"
  if [ -d "$1" ]; then
    quiet pg_ctl -D "$1" -l logfile stop
  else
    error 'Database server dir does not exist'
    exit 1
  fi
  success "Database server $(env_name "$1") stopped"
}

install_mamba() {
  include_conda

  info "Checking Conda version compatibility..."
  read -rp "$(color "$YELLOW" 'Would you like to update Conda before installing Mamba (recommended)? (y/n): ')" REPLY
  if [[ "$REPLY" =~ ^[Yy]$ ]]; then
    info "Updating Conda base environment (this may take a minute)..."
    quiet conda update -n base -y conda
    success "Conda updated successfully."
  else
    info "Skipping Conda update."
  fi

  if ! command -v mamba &>/dev/null; then
    info "Mamba not found — installing into base environment..."
    quiet conda install -n base -y -c conda-forge mamba
    success "Mamba installed successfully."
  else
    info "Mamba already installed."
  fi

  # --- Test mamba solver ---
  if ! quiet mamba info &>/dev/null; then
    warning "Mamba seems to have issues; falling back to Conda."
  else
    success "Mamba is working correctly."
  fi
}
