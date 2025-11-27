#!/bin/bash

TRAIN_CSV="data/train.csv"
VAL_CSV="data/val.csv"

BATCH=32
EPOCHS=5

# Choose your script:
# SCRIPT="train_without_attn.py"
SCRIPT="train_with_attn.py"

python3 $SCRIPT \
  --train_csv $TRAIN_CSV \
  --val_csv $VAL_CSV \
  --batch_size $BATCH \
  --epochs $EPOCHS \
  --output_dir "runs"
