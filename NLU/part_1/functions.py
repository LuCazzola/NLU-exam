# Utility
from conll import evaluate
import os
import numpy as np
import copy
from sklearn.metrics import classification_report
# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
# Logging
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import wandb

from model import ModelIAS
from utils import PAD_TOKEN, DEVICE


def train_step(model, data_loader, optimizer, criterion_slots, criterion_intents, clip=5):
    """
    Perform a single training step for the model using the provided data loader.

    Args:
        model (nn.Module): The model to be trained.
        data_loader (DataLoader): DataLoader providing training data.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        criterion_slots (nn.Module): Loss function for slot predictions.
        criterion_intents (nn.Module): Loss function for intent predictions.
        clip (float, optional): Gradient clipping value to prevent exploding gradients. Defaults to 5.

    Returns:
        list: A list of loss values recorded during the training step.
    """
    model.train()
    loss_array = []
    for sample in data_loader:
        optimizer.zero_grad() # Zeroing the gradient
        slots, intent = model(sample['utterances'], sample['slots_len'])
        loss_intent = criterion_intents(intent, sample['intents'])
        loss_slot = criterion_slots(slots, sample['y_slots'])
        loss = loss_intent + loss_slot # In joint training we sum the losses. 
                                       # Is there another way to do that?
        loss_array.append(loss.item())
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step() # Update the weights
    return loss_array


def eval_loop(model, data_loader, criterion_slots, criterion_intents, lang):
    """
    Evaluate the model on the provided data loader and compute metrics.

    Args:
        model (nn.Module): The model to be evaluated.
        data_loader (DataLoader): DataLoader providing evaluation data.
        criterion_slots (nn.Module): Loss function for slot predictions.
        criterion_intents (nn.Module): Loss function for intent predictions.
        lang (object): Language object containing id2intent and id2slot mappings.

    Returns:
        tuple: A tuple containing:
            - dict: Evaluation results for slots.
            - dict: Classification report for intents.
            - list: List of loss values recorded during the evaluation.
    """
    loss_array = []
    ref_intents = []
    hyp_intents = []
    ref_slots = []
    hyp_slots = []

    model.eval()
    #softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data_loader:
            slots, intents = model(sample['utterances'], sample['slots_len'])
            loss_intent = criterion_intents(intents, sample['intents'])
            loss_slot = criterion_slots(slots, sample['y_slots'])
            loss = loss_intent + loss_slot 
            loss_array.append(loss.item())
            # Intent inference
            # Get the highest probable class
            out_intents = [lang.id2intent[x] for x in torch.argmax(intents, dim=1).tolist()] 
            gt_intents = [lang.id2intent[x] for x in sample['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)
            
            # Slot inference 
            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                length = sample['slots_len'].tolist()[id_seq]
                utt_ids = sample['utterance'][id_seq][:length].tolist()
                gt_ids = sample['y_slots'][id_seq].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
                utterance = [lang.id2word[elem] for elem in utt_ids]
                to_decode = seq[:length].tolist()
                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                hyp_slots.append(tmp_seq)
    try:            
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        # Sometimes the model predicts a class that is not in REF
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        results = {"total":{"f":0}}
        
    report_intent = classification_report(ref_intents, hyp_intents, zero_division=False, output_dict=True)
    return results, report_intent, loss_array


def train_loop(model, train_loader, val_loader, optimizer, criterion_slots, criterion_intents, lang, n_epochs=200, clip=5, patience=5):
    """
    Train the model over multiple epochs and perform evaluation with early stopping.

    Args:
        model (nn.Module): The model to be trained.
        train_loader (DataLoader): DataLoader providing training data.
        val_loader (DataLoader): DataLoader providing validation data.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        criterion_slots (nn.Module): Loss function for slot predictions.
        criterion_intents (nn.Module): Loss function for intent predictions.
        lang (object): Language object containing id2intent and id2slot mappings.
        n_epochs (int, optional): Number of epochs to train. Defaults to 200.
        clip (float, optional): Gradient clipping value to prevent exploding gradients. Defaults to 5.
        patience (int, optional): Number of epochs to wait for improvement before stopping early. Defaults to 5.

    Returns:
        nn.Module: The best model obtained during training.
    """
    losses_train = []
    losses_val = []
    sampled_epochs = []
    best_model = None
    best_f1 = 0
    for epoch in tqdm(range(1,n_epochs+1)):
        loss = train_step(model, train_loader, optimizer, criterion_slots, criterion_intents, clip=clip)

        if epoch % 5 == 0: # We check the performance every 5 epochs
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())
            results_val, intent_res, loss_val = eval_loop(model, val_loader, criterion_slots, criterion_intents, lang)
            losses_val.append(np.asarray(loss_val).mean())

            if wandb.run is not None:
                wandb.log({"val Slot F1": results_val['total']['f'], "val Intent acc": intent_res['accuracy']}, step=epoch)
                wandb.log({"train loss": losses_train[-1]}, step=epoch)

            # For decreasing the patience you can also use the average between slot f1 and intent accuracy
            if results_val['total']['f'] >= best_f1:
                best_f1 = results_val['total']['f']
                best_model = copy.deepcopy(model).to('cpu')
                patience = 3
            else:
                patience -= 1
            if patience <= 0: # Early stopping with patience
                break

    return best_model


"""
INITIALIZATION FUNCTIONS
"""

def init_weights(mat):
    """
    Initialize weights for the model parameters using specific schemes for different layer types.

    Args:
        mat (nn.Module): The model or module whose weights are to be initialized.
    """
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


def init_components (args, lang) :
    """
    Initialize model components including the model, optimizer, and loss functions.

    Args:
        args (argparse.Namespace): Command line arguments containing configuration parameters.
        lang (object): Language object containing slot and intent mappings.

    Returns:
        tuple: A tuple containing:
            - nn.Module: Initialized model.
            - torch.optim.Optimizer: Initialized optimizer.
            - nn.Module: Loss function for slot predictions.
            - nn.Module: Loss function for intent predictions.
    """
    model = ModelIAS(args.emb_size, args.hid_size, len(lang.slot2id), len(lang.intent2id), len(lang.word2id),
        n_layer=args.n_layers,
        pad_index=PAD_TOKEN,
        bidirectional=args.bidirectional,
        dropout_enable=args.dropout_enable,
        emb_dropout=args.emb_dropout,
        hid_dropout=args.hid_dropout,
        out_dropout=args.out_dropout
    ).to(DEVICE)
    
    # Load model weights if the path is passed as parameter
    # Oterwise initialize base weights
    if args.load_checkpoint is not None :
        model.load_state_dict(torch.load(args.load_checkpoint, map_location=torch.device(DEVICE)))
    else :
        model.apply(init_weights)

    if args.optimizer_type == 'AdamW' :
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    elif args.optimizer_type == 'SGD' :
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    else :
        raise ValueError("Unsupported optimizer type.\n   - available choices : ASG, Adam, AdamW")

    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss()

    return model, optimizer, criterion_slots, criterion_intents

'''
UTILITY
'''

def save_model(model, filename='model.pt'):
    """
    Save the model's state dictionary to a file.

    Args:
        model (nn.Module): The model to be saved.
        filename (str, optional): Name of the file to save the model state. Defaults to 'model.pt'.
    """
    path = os.path.join("bin", filename)
    torch.save(model.state_dict(), path)


def get_num_parameters(model):
    """
    Calculate the total and trainable parameters of the model.

    Args:
        model (nn.Module): The model whose parameters are to be calculated.

    Returns:
        tuple: A tuple containing:
            - int: Total number of parameters.
            - int: Number of trainable parameters.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def get_args():
    """
    Parse and return command line arguments for model configuration and training.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Model parameters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model args
    parser.add_argument("--emb_size", type=int, default=300, required=False, help="size of word embedding")
    parser.add_argument("--hid_size", type=int, default=300, required=False, help="width of recurrent layers")
    parser.add_argument("--n_layers", type=int, default=1, required=False, help="number of stacked recurrent layers")
    parser.add_argument("--emb_dropout", type=float, default=0.1, required=False, help="embedding layer dropout probability (requires dropout enabled to function)")
    parser.add_argument("--hid_dropout", type=float, default=0.1, required=False, help="hidden recurrent layers dropout probability (requires dropout enabled to function)")
    parser.add_argument("--out_dropout", type=float, default=0.1, required=False, help="rnn output layer dropout probability (requires dropout enabled to function)")

    # Task specific flags
    parser.add_argument("--dropout_enable", default=False, action="store_true", required=False, help="enables dropout")
    parser.add_argument("--bidirectional", default=False, action="store_true", required=False, help="enables bidirectionality in LSTM cell")

    # Training/Inference related args
    parser.add_argument("--lr", type=float, default=0.0001, required=False, help="learning rate")
    parser.add_argument("--n_epochs", type=int, default=200, required=False, help="number of epochs")
    parser.add_argument("--runs", type=int, default=5, required=False, help="number of runs trainings to make before plotting results")
    parser.add_argument("--train_bsize", type=int, default=128, required=False, help="training set batch size")
    parser.add_argument("--val_bsize", type=int, default=64, required=False, help="validation set batch size")
    parser.add_argument("--test_bsize", type=int, default=64, required=False, help="test set batch size")
    parser.add_argument("--optimizer_type", type=str, default="Adam", required=False, help="type of optimizer {SGD, Adam, AdamW}")

    # Additional control flow arguments
    parser.add_argument("--load_checkpoint", type=str, default=None, required=False, help="path to weight checkpoint to load")
    parser.add_argument("--test_only",  default=False, action="store_true", required=False, help="avoid training the model. Perform inference on test set only")
    parser.add_argument("--save_model", default=False, action="store_true", required=False, help="store the model in 'model_bin/'")
    parser.add_argument("--enable_logger", default=False, action="store_true", required=False, help="log to wandb")

    return parser.parse_args()


def init_logger(args):
    """
    Initialize a logger using Weights & Biases to track training and evaluation metrics.

    Args:
        args (argparse.Namespace): Command line arguments containing configuration parameters.
    """
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="NLU-part-2.1",

        # track hyperparameters and run metadata
        config={
        "learning_rate": args.lr,
        "embedding_size" : args.emb_size,
        "hidden_size" : args.hid_size,
        "n_layers" : args.n_layers,
        "dataset": "ATIS",
        "optimizer" : args.optimizer_type,
        "epochs": args.n_epochs,
        "bidirectional" : args.bidirectional,
        "dropout_enable" : args.dropout_enable,
        "emb_dropout" : args.emb_dropout,
        "hid_dropout" : args.hid_dropout,
        "out_dropout" : args.out_dropout
        }
    )