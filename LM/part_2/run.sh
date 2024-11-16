#!/bin/bash

### VARIABLES ###

# Parameters for the model
EMB_SIZE=300               # Size of the word embedding
HID_SIZE=300               # Width of recurrent layers
RECLAYER_TYPE="LSTM"        # (LSTM, GRU) : Type of recurrent cell
N_LAYERS=2                 # number of stacked recurrent layers
EMB_DROPOUT=0.1            # embedding layer dropout probability
HID_DROPOUT=0.25           # hidden layers dropout probability
OUT_DROPOUT=0.1            # rnn output layer dropout probability

# assignment related flags
WEIGHT_TYING=true         # (true/false) : Whether to applying weight tying to embedding/output layers
VAR_DROPOUT=true          # (true/false) : Whether to add variational dropout instead of standard dropout
NMT_AVSGD_ENABLED=true    # (true/false) : Whether to add non-monotonically-triggered AvSGD (requires SGD set as 'OPTIMIZER_TYPE')

# Training settings
LR=5                      # Learning rate
N_EPOCHS=100               # Number of epochs
OPTIMIZER_TYPE="SGD"       # (SGD, AdamW) : Type of optimizer
TRAIN_BSIZE=64             # Training set batch size
VAL_BSIZE=128              # Validation set batch size
TEST_BSIZE=128             # Test set batch size 

# Additional control flow arguments
LOAD_CHECKPOINT="None"     # Path to weight checkpoint to load (set to "None" if not used)
TEST_ONLY=false            # (true/false) : Whether to perform inference on test set only or train the model + perform testing
SAVE_MODEL=false           # (true/false) : Whether to save the model
ENABLE_LOGGER=false        # (true/false) : Whether to enable logging to wandb


### COMMAND COMPOSITION ###

# Construct the command with the arguments
CMD="python main.py"

# Include variables
CMD+=" --emb_size $EMB_SIZE"
CMD+=" --hid_size $HID_SIZE"
CMD+=" --n_layers $N_LAYERS"
CMD+=" --emb_dropout $EMB_DROPOUT"
CMD+=" --hid_dropout $HID_DROPOUT"
CMD+=" --out_dropout $OUT_DROPOUT"
CMD+=" --lr $LR"
CMD+=" --n_epochs $N_EPOCHS"
CMD+=" --train_bsize $TRAIN_BSIZE"
CMD+=" --val_bsize $VAL_BSIZE"
CMD+=" --test_bsize $TEST_BSIZE"
CMD+=" --optimizer_type $OPTIMIZER_TYPE"
CMD+=" --recLayer_type $RECLAYER_TYPE"

# Include flags
if [ "$WEIGHT_TYING" = true ]; then
  CMD+=" --weight_tying"
fi
if [ "$VAR_DROPOUT" = true ]; then
  CMD+=" --var_dropout"
fi
if [ "$NMT_AVSGD_ENABLED" = true ]; then
  CMD+=" --nmt_AvSGD_enabled"
fi
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


