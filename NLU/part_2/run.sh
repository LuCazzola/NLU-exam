#!/bin/bash

### VARIABLES ###

# Parameters for the model
FINETUNE_BERT=true               # (true/false) : Whether to fine-tune the BERT model or keep it Frozen
BERT_VERSION="bert-base-uncased" # (bert-base-uncased, bert-large-uncased, bert-base-cased, bert-large-cased) 

MERGER_ENABLE=true               # (true/false) enables the subtoken merger
NUM_HEADS=4                      # Number of attention heads

DROPOUT_ENABLE=true              # (true/false) enables dropout
INT_DROPOUT=0.1                  # intents dropout probability (applied before classifier)
SLOT_DROPOUT=0.1                 # slots dropout probability (applied before classifier)

# Training settings
OPTIMIZER_TYPE="Adam"       # (SGD, Adam, AdamW) : Type of optimizer
LR=0.0001                   # Learning rate
N_EPOCHS=30                 # Number of epochs
RUNS=3                      # how many times to run the model training
MOMENTUM=0.9                # apply momentum to the optimizer
TRAIN_BSIZE=128             # Training set batch size
VAL_BSIZE=64                # Validation set batch size
TEST_BSIZE=64               # Test set batch size 

# Additional control flow arguments
LOAD_CHECKPOINT="None"     # Path to weight checkpoint to load (set to "None" if not used)
TEST_ONLY=false            # (true/false) : Whether to perform inference on test set only or train the model + perform testing
SAVE_MODEL=false            # (true/false) : Whether to save the model
ENABLE_LOGGER=false        # (true/false) : Whether to enable logging to wandb

### COMMAND COMPOSITION ###

# Construct the command with the arguments
CMD="python main.py"

# Include variables
CMD+=" --bert_version $BERT_VERSION"
CMD+=" --num_heads $NUM_HEADS"
CMD+=" --int_dropout $INT_DROPOUT"
CMD+=" --slot_dropout $SLOT_DROPOUT"
CMD+=" --lr $LR"
CMD+=" --n_epochs $N_EPOCHS"
CMD+=" --runs $RUNS"
CMD+=" --train_bsize $TRAIN_BSIZE"
CMD+=" --val_bsize $VAL_BSIZE"
CMD+=" --test_bsize $TEST_BSIZE"
CMD+=" --optimizer_type $OPTIMIZER_TYPE"
CMD+=" --momentum $MOMENTUM"

# Include flags

if [ "$FINETUNE_BERT" = true ]; then
  CMD+=" --finetune_bert"
fi
if [ "$DROPOUT_ENABLE" = true ]; then
  CMD+=" --dropout_enable"
fi
if [ "$MERGER_ENABLE" = true ]; then
  CMD+=" --merger_enable"
fi
# Additional control flow arguments
if [ "$LOAD_CHECKPOINT" != "None" ]; then
  CMD+=" --load_checkpoint $LOAD_CHECKPOINT"
fi
if [ "$SAVE_MODEL" = true ]; then
  CMD+=" --save_model"
fi
if [ "$ENABLE_LOGGER" = true ]; then
  CMD+=" --enable_logger"
fi
if [ "$TEST_ONLY" = true ]; then
  CMD+=" --test_only"
fi

# Print the command for debugging
echo "Executing => $CMD"
# Execute the command
eval $CMD


