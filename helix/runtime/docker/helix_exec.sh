#!/usr/bin/env bash
set -euo pipefail

CACHE_ROOT="${HELIX_CACHE_ROOT:-/helix-cache}"
VENV_DIR="${HELIX_VENV_DIR:-${CACHE_ROOT}/venv}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${CACHE_ROOT}/pip}"
NPM_CACHE_DIR="${NPM_CONFIG_CACHE:-${CACHE_ROOT}/npm}"
NPM_PREFIX_DIR="${NPM_CONFIG_PREFIX:-${CACHE_ROOT}/npm-global}"
HOME_DIR="${HOME:-${CACHE_ROOT}/home}"

mkdir -p "${CACHE_ROOT}" "${PIP_CACHE_DIR}" "${NPM_CACHE_DIR}" "${NPM_PREFIX_DIR}" "${HOME_DIR}"

RESET_VENV=0
if [ ! -x "${VENV_DIR}/bin/python" ]; then
  RESET_VENV=1
elif ! grep -q '^include-system-site-packages = true$' "${VENV_DIR}/pyvenv.cfg" 2>/dev/null; then
  # Upgrade old cache volumes created before the browser/runtime packages were
  # exposed through the persistent venv.
  RESET_VENV=1
fi

if [ "${RESET_VENV}" = "1" ]; then
  rm -rf "${VENV_DIR}"
  python -m venv --system-site-packages "${VENV_DIR}"
fi

export HOME="${HOME_DIR}"
export VIRTUAL_ENV="${VENV_DIR}"
export PATH="${VENV_DIR}/bin:${NPM_PREFIX_DIR}/bin:${PATH}"
export PIP_CACHE_DIR
export NPM_CONFIG_CACHE="${NPM_CACHE_DIR}"
export NPM_CONFIG_PREFIX="${NPM_PREFIX_DIR}"

exec "$@"
