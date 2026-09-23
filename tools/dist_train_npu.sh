#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:?configuration file is required}
NPUS=${2:-1}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}

case "$NPUS" in
    ''|*[!0-9]*)
        echo "NPUS must be a positive integer, got: $NPUS" >&2
        exit 2
        ;;
esac
if [ "$NPUS" -lt 1 ]; then
    echo "NPUS must be >= 1" >&2
    exit 2
fi

PYTHONPATH="$(dirname "$0")/..:${PYTHONPATH:-}" \
python -m torch.distributed.run \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --nproc_per_node="$NPUS" \
    --master_port="$MASTER_PORT" \
    "$(dirname "$0")/train.py" "$CONFIG" \
    --launcher pytorch "${@:3}"
