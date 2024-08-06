"""
Module for training and evaluating a recurrent neural network (RNN) model.
"""

# Utility libraries
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
# pytorch modules
import torch.optim as optim
# Logging
import argparse
import wandb
# Cross project references
from utils import *
from model import *

"""
MODEL TRAINING / INFERENCE
"""

def train_step(model, data_loader, optimizer, criterion, clip):
    """
    Perform a single training step on the given model using the provided data loader, optimizer, and criterion.
    
    Args:
        model (torch.nn.Module): The RNN model to be trained.
        data_loader (torch.utils.data.DataLoader): DataLoader providing the training data.
        optimizer (torch.optim.Optimizer): Optimizer for updating the model parameters.
        criterion (torch.nn.Module): Loss function.
        clip (float): Gradient clipping value.
    
    Returns:
        tuple: Perplexity and average loss of the training step.
    """
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
        # clip the gradient to avoid explosioning gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step() # Update the weights
    
    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    loss_to_return = sum(loss_array) / sum(number_of_tokens)

    return ppl, loss_to_return


def train_loop(model, train_loader, val_loader, optimizer, criterion_train, criterion_val, n_epochs=100, patience=3, clip=5):
    """
    Execute the training loop for the specified number of epochs, incorporating early stopping based on validation performance.
    
    Args:
        model (torch.nn.Module): The RNN model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader providing the training data.
        val_loader (torch.utils.data.DataLoader): DataLoader providing the validation data.
        optimizer (torch.optim.Optimizer): Optimizer for updating the model parameters.
        criterion_train (torch.nn.Module): Loss function for training.
        criterion_val (torch.nn.Module): Loss function for validation.
        n_epochs (int, optional): Number of epochs to train the model. Default is 100.
        patience (int, optional): Number of epochs with no improvement after which training will be stopped. Default is 3.
        clip (float, optional): Gradient clipping value. Default is 5.
    
    Returns:
        torch.nn.Module: The best model based on validation performance.
    """
    losses_train = []
    losses_val = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None

    pbar = tqdm(range(1, n_epochs+1))
    for epoch in pbar:
        
        ppl_train, loss_train = train_step(model, train_loader, optimizer, criterion_train, clip)

        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss_train).mean())
            ppl_val, loss_val = eval_loop(model, val_loader, criterion_val)
            losses_val.append(np.asarray(loss_val).mean())
            pbar.set_description("PPL: %f" % ppl_val)

            # Log Train - Validation sets metrics if wandb logger is available
            if wandb.run is not None:
                wandb.log({"train perplexity": ppl_train, "train loss":losses_train[-1]}, step=epoch)
                wandb.log({"val perplexity": ppl_val, "val loss":losses_val[-1]}, step=epoch)

            if  ppl_val < best_ppl: # the lower, the better
                best_ppl = ppl_val
                best_model = copy.deepcopy(model).to('cpu')
                patience = 3
            else:
                patience -= 1
                
            if patience <= 0: # Early stopping with patience
                break
        
    return best_model

def eval_loop(model, data_loader, criterion):
    """
    Evaluate the model on the given data loader using the specified criterion.
    
    Args:
        model (torch.nn.Module): The RNN model to be evaluated.
        data_loader (torch.utils.data.DataLoader): DataLoader providing the evaluation data.
        criterion (torch.nn.Module): Loss function for evaluation.
    
    Returns:
        tuple: Perplexity and average loss of the evaluation step.
    """
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
    """
    Initialize the weights of the model using Xavier uniform initialization for input-hidden weights and orthogonal initialization for hidden-hidden weights.
    
    Args:
        mat (torch.nn.Module): The RNN model whose weights need to be initialized.
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


def init_modelComponents (args, lang) :
    """
    Initialize the model, optimizer, and loss functions based on the provided arguments and language settings.
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        lang (object): Language object containing word-to-id mappings.
    
    Returns:
        tuple: Initialized model, optimizer, training criterion, and evaluation criterion.
    """
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
        dropout_enabled=args.dropout_enabled).to(DEVICE)

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
    """
    Save the model's state dictionary to a specified file.
    
    Args:
        model (torch.nn.Module): The RNN model to be saved.
        filename (str, optional): Name of the file to save the model. Default is 'model.pt'.
    """
    path = 'bin/'+filename
    torch.save(model.state_dict(), path)

def get_args():
    """
    Parse and return the command-line arguments for model configuration and training.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Model parameters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model args
    parser.add_argument("--emb_size", type=int, default=300, required=False, help="size of word embedding")
    parser.add_argument("--hid_size", type=int, default=200, required=False, help="width of recurrent layers")
    parser.add_argument("--recLayer_type", type=str, default="vanilla", required=False, help="type of recurrent cell {vanilla, LSTM, GRU}")
    parser.add_argument("--n_layers", type=int, default=1, required=False, help="number of stacked recurrent layers")
    parser.add_argument("--emb_dropout", type=float, default=0.1, required=False, help="embedding layer dropout probability (requires dropout enabled to function)")
    parser.add_argument("--hid_dropout", type=float, default=0.1, required=False, help="hidden recurrent layers dropout probability (requires dropout enabled to function)")
    parser.add_argument("--out_dropout", type=float, default=0.1, required=False, help="rnn output layer dropout probability (requires dropout enabled to function)")
    parser.add_argument("--dropout_enabled", default=False, action="store_true", required=False, help="adds 2 dropout layers to the model: one after the embedding layer, one before the last linear layer")

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
    """
    Initialize the Weights and Biases (wandb) logger for tracking experiments.
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="NLU-part1",

        # track hyperparameters and run metadata
        config={
        "learning_rate": args.lr,
        "embedding_size" : args.emb_size,
        "hidden_size" : args.hid_size,
        "architecture": args.RNN_type,
        "dataset": "PennTreeBank",
        "epochs": args.n_epochs,
        }
    )
