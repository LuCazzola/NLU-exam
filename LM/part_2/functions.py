"""
Module for training and evaluating a recurrent neural network (RNN) model.
"""

# Utility libraries
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import numpy as np
# pytorch modules
import torch
import torch.nn as nn
import torch.optim as optim
# Logging
import argparse
import wandb
# Cross project references
from model import LM_RNN
from utils import DEVICE

"""
MODEL TRAINING / INFERENCE
"""

def train_step(model, data_loader, optimizer, criterion, clip):

    model.train()
    loss_array = []
    number_of_tokens = []
    
    for sample in data_loader:
        optimizer.zero_grad() # Zeroing the gradient
        output = model(sample['source'])
        loss = criterion(output, sample['target'])
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])
        loss.backward() # Compute the gradient, deleting the computational graph
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  # clip the gradient to avoid explosioning gradients
        optimizer.step() # Update the weights
    
    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    loss_to_return = sum(loss_array) / sum(number_of_tokens)

    return ppl, loss_to_return


def train_loop(model, train_loader, val_loader, optimizer, criterion_train, criterion_val, n_epochs=100, patience=3, clip=5, ppl_milestones=[140, 120, 100], nmt_AvSGD_enabled=False, non_monotone_int=5):

    ppls_train = []
    ppls_val =[]
    sampled_epochs = []
    
    best_ppl = math.inf
    curr_patience = patience
    best_model = None
    nm_trigger_flag = False      # Non-Monotonic trigger flag for ASGD
    trigger_point = None

    pbar = tqdm(range(1, n_epochs+1))
    for epoch in pbar:
        
        ppl_train, loss_train = train_step(model, train_loader, optimizer, criterion_train, clip)

        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            # Store train metric
            ppls_train.append(ppl_train)

            model_params_bckp = {}
            if nmt_AvSGD_enabled and nm_trigger_flag :
                # substitute current parameter configuration with running average on inference (if ASGD triggered)
                for p in model.parameters():
                    model_params_bckp[p] = p.data.clone()
                    p.data = optimizer.state[p]['ax'].clone()

            # Validation set inference
            ppl_val, loss_val = eval_loop(model, val_loader, criterion_val)
            ppls_val.append(ppl_val)

            if nm_trigger_flag :
                pbar.set_description(f"> PPL: {ppl_val}, patience: {curr_patience}, AvSGD triggered at: {trigger_point}")
            else :
                pbar.set_description(f"> PPL: {ppl_val}, Patience: {curr_patience}")
            # Log Train - Validation sets metrics if wandb logger is available
            if wandb.run is not None:
                wandb.log({"train perplexity": ppl_train, "train loss":loss_train}, step=sampled_epochs[-1])
                wandb.log({"val perplexity": ppl_val, "val loss":loss_train}, step=sampled_epochs[-1])

            if  ppl_val < best_ppl:
                # if ppl lowers store current weight conf. as best model
                best_ppl = ppl_val
                best_model = copy.deepcopy(model).to('cpu')
                curr_patience = patience
            elif not nmt_AvSGD_enabled or nm_trigger_flag:
                # when 'nmt_AvSGD_enabled' patience decreases only after the trigger happens
                # otherwise it always decreses
                curr_patience -= 1

            if curr_patience <= 0: # Early stopping with patience
                break
        
            if nmt_AvSGD_enabled and nm_trigger_flag :
                # restore prev. weights (if ASGD triggered)
                for p in model.parameters():
                    p.data = model_params_bckp[p].clone()

            # when a perplexity milestone is reachd divide learning rate by a costant factor
            if len(ppl_milestones) != 0:
                if ppl_milestones[0] >= ppl_val :
                    optimizer.param_groups[0]['lr'] /= 10.
                    _ = ppl_milestones.pop(0)

            # Trigger condition to switch optimizer
            if nmt_AvSGD_enabled and type(optimizer).__name__ == 'SGD' and not nm_trigger_flag and epoch > non_monotone_int and ppl_val > min(ppls_val[:-non_monotone_int]) :
                nm_trigger_flag = True
                trigger_point = sampled_epochs[-1]
                optimizer = torch.optim.ASGD(model.parameters(), lr=optimizer.param_groups[0]['lr'], t0=0, lambd=0., weight_decay=optimizer.param_groups[0]['weight_decay'])

                if wandb.run is not None:
                    wandb.log({"AvSGD trigger point": trigger_point}, step=epoch)
        
    return best_model

def eval_loop(model, data_loader, criterion):

    model.eval()
    loss_to_return = []
    loss_array = []
    number_of_tokens = []
    # softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data_loader:
            output = model(sample['source'])
            loss = criterion(output, sample['target'])
            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])
            
    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    return ppl, loss_to_return


"""
INITIALIZATION FUNCTIONS
"""

def init_weights(mat):

    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)


def init_modelComponents (args, lang) :

    # Load the model
    model = LM_RNN(
        args.emb_size,
        args.hid_size,
        len(lang.word2id),
        pad_index=lang.word2id["<pad>"],
        emb_dropout=args.emb_dropout,
        hid_dropout=args.hid_dropout,
        out_dropout=args.out_dropout,
        n_layers=args.n_layers,
        recLayer_type=args.recLayer_type,
        var_dropout=args.var_dropout,
        weight_tying=args.weight_tying).to(DEVICE)
    
    # Load model weights are passed as parameters load them
    # Oterwise initialize base weights
    if args.load_checkpoint is not None :
        model.load_state_dict(torch.load(args.load_checkpoint))
    else :
        model.apply(init_weights)

    # Define the optimizer
    if args.optimizer_type == 'SGD' :
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer_type == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
    else :
        print("Unsupported optimizer type.\n   - available choices : {SGD, AdamW}")
        exit()
            
    # Set Training loss
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    # Set validation & test loss
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

    return model, optimizer, criterion_train, criterion_eval


"""
UTILITY
"""

def save_model(model, filename='model.pt'):

    path = 'bin/'+filename
    torch.save(model.state_dict(), path)


def get_num_parameters(model):
    return sum(p.numel() for p in model.parameters())
    

def get_args():

    parser = argparse.ArgumentParser(
        description="Model parameters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model args
    parser.add_argument("--emb_size", type=int, default=300, required=False, help="size of word embedding")
    parser.add_argument("--hid_size", type=int, default=200, required=False, help="width of recurrent layers")
    parser.add_argument("--recLayer_type", type=str, default="LSTM", required=False, help="type of recurrent cell {LSTM}")
    parser.add_argument("--n_layers", type=int, default=1, required=False, help="number of stacked recurrent layers")
    parser.add_argument("--emb_dropout", type=float, default=0.1, required=False, help="embedding layer dropout probability (requires dropout enabled to function)")
    parser.add_argument("--hid_dropout", type=float, default=0.1, required=False, help="hidden recurrent layers dropout probability (requires dropout enabled to function)")
    parser.add_argument("--out_dropout", type=float, default=0.1, required=False, help="rnn output layer dropout probability (requires dropout enabled to function)")
    # assignment related flags
    parser.add_argument("--weight_tying", default=False, action="store_true", required=False, help="enables weight tying between embedding and output layer")
    parser.add_argument("--var_dropout", default=False, action="store_true", required=False, help="adds variational dropout to the model")
    parser.add_argument("--nmt_AvSGD_enabled", default=False, action="store_true", required=False, help="adds non-monotonically-triggered AvSGD (requires SGD set as optimizer)")

    # Training/Inference related args
    parser.add_argument("--lr", type=float, default=0.0001, required=False, help="learning rate")
    parser.add_argument("--n_epochs", type=int, default=100, required=False, help="number of epochs")
    parser.add_argument("--train_bsize", type=int, default=64, required=False, help="training set batch size")
    parser.add_argument("--val_bsize", type=int, default=128, required=False, help="validation set batch size")
    parser.add_argument("--test_bsize", type=int, default=128, required=False, help="test set batch size")
    parser.add_argument("--optimizer_type", type=str, default="SGD", required=False, help="type of optimizer {SGD, AdamW}")

    # Additional control flow arguments
    parser.add_argument("--load_checkpoint", type=str, default=None, required=False, help="path to weight checkpoint to load")
    parser.add_argument("--test_only",  default=False, action="store_true", required=False, help="avoid training the model. Perform inference on test set only")
    parser.add_argument("--save_model", default=False, action="store_true", required=False, help="store the model in 'model_bin/'")
    parser.add_argument("--enable_logger", default=False, action="store_true", required=False, help="log to wandb")

    return parser.parse_args()



def init_logger(args):
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="NLU-part1",

        # track hyperparameters and run metadata
        config={
        "learning_rate": args.lr,
        "embedding_size" : args.emb_size,
        "hidden_size" : args.hid_size,
        "architecture": args.recLayer_type,
        "dataset": "PennTreeBank",
        "epochs": args.n_epochs,
        }
    )
