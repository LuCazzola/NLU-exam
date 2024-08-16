# Task description - Lab 4, Part 2
Starting from the LM_RNN in which you replaced the RNN with a LSTM model, apply the following regularisation techniques:

* Weight Tying
* Variational Dropout (no DropConnect)
* Non-monotonically Triggered AvSGD

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
python main.py --emb_size 300 --hid_size 300 --n_layers 2 --emb_dropout 0.1 --hid_dropout 0.25 --out_dropout 0.1 --lr 5 --n_epochs 100 --train_bsize 64 --val_bsize 128 --test_bsize 128 --optimizer_type SGD --recLayer_type LSTM --weight_tying --var_dropout --nmt_AvSGD_enabled
```
<br>

## Test example
To run inference only on the provided weights :
```
python main.py --emb_size 300 --hid_size 300 --n_layers 2 --emb_dropout 0.1 --hid_dropout 0.25 --out_dropout 0.1 --lr 5 --n_epochs 100 --train_bsize 64 --val_bsize 128 --test_bsize 128 --optimizer_type SGD --recLayer_type LSTM --weight_tying --var_dropout --nmt_AvSGD_enabled --load_checkpoint bin/best_model.pt --test_only
```
