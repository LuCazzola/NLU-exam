import os

try :


    # LSTM with Adamw
    os.system("python main.py --emb_size 300 --hid_size 300 --n_layers 1 --emb_dropout 0.1 --hid_dropout 0.1 --out_dropout 0.1 --lr 0.0005 --n_epochs 200 --runs 4 --train_bsize 128 --val_bsize 64 --test_bsize 64 --optimizer_type AdamW --enable_logger")

    os.system("python main.py --emb_size 300 --hid_size 300 --n_layers 1 --emb_dropout 0.1 --hid_dropout 0.1 --out_dropout 0.1 --lr 0.0005 --n_epochs 200 --runs 4 --train_bsize 128 --val_bsize 64 --test_bsize 64 --optimizer_type AdamW --bidirectional --enable_logger")

    os.system("python main.py --emb_size 300 --hid_size 300 --n_layers 1 --emb_dropout 0.1 --hid_dropout 0.1 --out_dropout 0.1 --lr 0.0005 --n_epochs 200 --runs 4 --train_bsize 128 --val_bsize 64 --test_bsize 64 --optimizer_type AdamW --dropout_enable --bidirectional --enable_logger")

except KeyboardInterrupt:
    print("KEYBOARD INTERRUPT")
    exit() 










