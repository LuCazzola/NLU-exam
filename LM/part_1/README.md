# Task description - Lab 4, Part 1
In this, you have to modify the baseline LM_RNN by adding a set of techniques that might improve the performance. In this, you have to add one modification at a time incrementally.
For each of your experiments, you have to print the performance expressed with Perplexity (PPL).

* Replace RNN with a Long-Short Term Memory (LSTM) network
* Add two dropout layers
  * one after the embedding layer
  * one before the last linear layer
* Replace SGD with AdamW

## Note for professors
The project uses **wandb** as logger to store the metrics.
If you've already installed the requirements of the [**course labs**](https://github.com/BrownFortress/NLU-2024-Labs), simply install the dependancy for wandb
```
pip install wandb
```

<br><hr><br>

# Usage

A bash script is available to make setting command line arguments easier setting easier :
```
chmod +x run.sh
./run.sh
```
Otherwise interact directly with the script. To check available options :
```
python main.py --help
```
<br>

## Train example
Training example
```
python main.py --emb_size 300 --hid_size 300 --n_layers 1 --recLayer_type LSTM --dropout_enabled --lr 0.0001 --n_epochs 100 --emb_dropout 0.1 --hid_dropout 0.1 --out_dropout 0.1 --train_bsize 64 --val_bsize 128 --test_bsize 128 --optimizer_type AdamW
```
<br>

## Test example
To run inference only on the provided weights :
```
python main.py --emb_size 300 --hid_size 300 --n_layers 1 --recLayer_type LSTM --dropout_enabled --lr 0.0001 --n_epochs 100 --emb_dropout 0.1 --hid_dropout 0.1 --out_dropout 0.1 --train_bsize 64 --val_bsize 128 --test_bsize 128 --optimizer_type AdamW --load_checkpoint bin/best_model.pt --test_only
```
