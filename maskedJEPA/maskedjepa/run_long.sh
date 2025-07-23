#!/bin/bash

LOGFILE="log_jepa1.log"

CMD="python -m app.main --fname configs/pretrain/vith16.yaml"

echo "Running: $CMD"
echo "Logging output to: $LOGFILE"

$CMD 2>&1 | tee "$LOGFILE"
