# Task description - Lab 5, Part 1
As for LM project, you have to apply these two modifications incrementally. Also in this case you may have to play with the hyperparameters and optimizers to improve the performance.
Modify the baseline architecture Model IAS by:

* Adding bidirectionality
* Adding dropout layer

-> Intent classification: accuracy
-> Slot filling: F1 score with conll
-> Dataset to use: ATIS

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
python main.py --emb_size 300 --hid_size 300 --n_layers 1 --emb_dropout 0.1 --hid_dropout 0.1 --out_dropout 0.1 --lr 5 --n_epochs 200 --runs 4 --train_bsize 128 --val_bsize 64 --test_bsize 64 --optimizer_type SGD --dropout_enable --bidirectional
```
<br>

## Test example
To run inference only on the provided weights :

(best intents)
```
python main.py --emb_size 300 --hid_size 300 --n_layers 1 --emb_dropout 0.1 --hid_dropout 0.1 --out_dropout 0.1 --lr 5 --n_epochs 200 --runs 4 --train_bsize 128 --val_bsize 64 --test_bsize 64 --optimizer_type SGD --dropout_enable --bidirectional --load_checkpoint bin/intents-best_model.pt --test_only
```
(best slots)
```
python main.py --emb_size 300 --hid_size 300 --n_layers 1 --emb_dropout 0.1 --hid_dropout 0.1 --out_dropout 0.1 --lr 5 --n_epochs 200 --runs 4 --train_bsize 128 --val_bsize 64 --test_bsize 64 --optimizer_type SGD --dropout_enable --bidirectional --load_checkpoint bin/slots-best_model.pt --test_only
```
