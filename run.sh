#!/bin/bash
set -euxo pipefail
umask 0022

source .venv/bin/activate

export TOKENIZERS_PARALLELISM="true"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

python -m rmgap.cli.main_eval "$@"
