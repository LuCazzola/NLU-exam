# Task description - Lab 5, Part 2
Adapt the code to fine-tune a pre-trained BERT model using a multi-task learning setting on intent classification and slot filling.
You can refer to this paper to have a better understanding of how to implement this [**Here**](https://arxiv.org/abs/1902.10909).
In this, one of the challenges of this is to handle the sub-tokenization issue.

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
python main.py --bert_version bert-base-uncased --num_heads 4 --int_dropout 0.1 --slot_dropout 0.1 --lr 0.0001 --n_epochs 30 --runs 3 --train_bsize 128 --val_bsize 64 --test_bsize 64 --optimizer_type Adam --momentum 0.9 --finetune_bert --dropout_enable --merger_enable
```
<br>

## Test example
To run inference only on the provided weights :

(best intents)
```
python main.py --bert_version bert-base-uncased --num_heads 4 --int_dropout 0.1 --slot_dropout 0.1 --lr 0.0001 --n_epochs 30 --runs 3 --train_bsize 128 --val_bsize 64 --test_bsize 64 --optimizer_type Adam --momentum 0.9 --finetune_bert --dropout_enable --merger_enable --load_checkpoint bin/intents-best_model.pt --test_only
```
(best slots)
```
python main.py --bert_version bert-base-uncased --num_heads 4 --int_dropout 0.1 --slot_dropout 0.1 --lr 0.0001 --n_epochs 30 --runs 3 --train_bsize 128 --val_bsize 64 --test_bsize 64 --optimizer_type Adam --momentum 0.9 --finetune_bert --dropout_enable --load_checkpoint bin/slots-best_model.pt --test_only
```
