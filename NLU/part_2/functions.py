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

torch.autograd.set_detect_anomaly(True)

def train_step(model, data_loader, optimizer, criterion_slots, criterion_intents, lang, clip=5):
    """
    Perform a single training step.

    Args:
        model (torch.nn.Module): The model to train.
        data_loader (DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        criterion_slots (nn.Module): Loss function for slot prediction.
        criterion_intents (nn.Module): Loss function for intent prediction.
        lang (Lang): Language object for token mappings.
        clip (float, optional): Gradient clipping value. Defaults to 5.

    Returns:
        list: List of loss values for each batch.
    """
    model.train()
    loss_array = []
    for sample in data_loader:
        optimizer.zero_grad() # Zeroing the gradient
        # Prepare the input for the model
        bert_input, bert_slots, special_tokens = set_bert_input_and_slots (sample, lang, model.tokenizer)
        subtoken_poses = get_subtok_poses(special_tokens, bert_input['input_ids'].shape[0])
        
        # forward input to model
        slots, intents, _ = model(bert_input, subtoken_poses)

        loss_intent = criterion_intents(intents, sample['intents'])
        loss_slot = criterion_slots(slots, bert_slots)
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
    Evaluate the model on validation or test data.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data_loader (DataLoader): DataLoader for evaluation data.
        criterion_slots (nn.Module): Loss function for slot prediction.
        criterion_intents (nn.Module): Loss function for intent prediction.
        lang (Lang): Language object for token mappings.

    Returns:
        dict: Evaluation results including slot F1 score and intent accuracy.
        dict: Classification report for intents.
        list: List of loss values for each batch.
        list: List of subtoken weights.
    """
    loss_array = []
    ref_intents = []
    hyp_intents = []
    ref_slots = []
    hyp_slots = []
    subtok_weights_array = []

    model.eval()
    #softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data_loader:
            # Prepare the input for the model
            bert_input, bert_slots, special_tokens = set_bert_input_and_slots (sample, lang, model.tokenizer)
            slots_eval_ignore = copy.deepcopy(special_tokens) # This is used to ignore the special tokens on the slot evaluation
            subtoken_poses = get_subtok_poses(special_tokens, bert_input['input_ids'].shape[0])

            # forward input to model
            slots, intents, subtok_weights = model(bert_input, subtoken_poses)
            subtok_weights_array.append(subtok_weights)

            loss_intent = criterion_intents(intents, sample['intents'])
            loss_slot = criterion_slots(slots, bert_slots)

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
                # Remove the ignored tokens
                seq = [elem for id_el, elem in enumerate(seq.tolist()) if id_el not in slots_eval_ignore[id_seq]]
                # Rest remains the same
                length = sample['slots_len'].tolist()[id_seq]
                utt_ids = sample['utterance'][id_seq][:length].tolist()
                gt_ids = sample['y_slots'][id_seq].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
                utterance = [lang.id2word[elem] for elem in utt_ids]
                to_decode = seq[:length]
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
    return results, report_intent, loss_array, subtok_weights_array


def train_loop(model, train_loader, val_loader, optimizer, criterion_slots, criterion_intents, lang, n_epochs=200, clip=5, patience=5):
    """
    Main training loop with validation.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        criterion_slots (nn.Module): Loss function for slot prediction.
        criterion_intents (nn.Module): Loss function for intent prediction.
        lang (Lang): Language object for token mappings.
        n_epochs (int, optional): Number of epochs. Defaults to 200.
        clip (float, optional): Gradient clipping value. Defaults to 5.
        patience (int, optional): Early stopping patience. Defaults to 5.

    Returns:
        torch.nn.Module: Best model after training.
    """
    losses_train = []
    losses_val = []
    sampled_epochs = []
    subtok_weights_epochs = []

    best_model = None
    best_f1 = 0
    for epoch in tqdm(range(1,n_epochs+1)):
        loss = train_step(model, train_loader, optimizer, criterion_slots, criterion_intents, lang, clip=clip)

        if epoch % 5 == 0 or epoch == 1:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())
            results_val, intent_res, loss_val, subtok_weights = eval_loop(model, val_loader, criterion_slots, criterion_intents, lang)
            subtok_weights_epochs.append(subtok_weights)
            losses_val.append(np.asarray(loss_val).mean())

            #print(f"[{epoch}] SUBTOK_WEIGHTS ", subtok_weights_epochs[-1][0])
            #print("==="*20)

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

def init_components (args, lang) :
    """
    Initialize model, optimizer, and loss functions.

    Args:
        args (argparse.Namespace): Command-line arguments.
        lang (Lang): Language object for token mappings.

    Returns:
        tuple: Model, optimizer, slot loss function, and intent loss function.
    """
    model = ModelIAS(len(lang.slot2id), len(lang.intent2id), len(lang.word2id),
        finetune_bert=args.finetune_bert,
        bert_version=args.bert_version,
        dropout_enable=args.dropout_enable,
        int_dropout=args.int_dropout,
        slot_dropout=args.slot_dropout,
        merger_enable=args.merger_enable,
        num_heads=args.num_heads,
    ).to(DEVICE)

    # Load model weights if the path is passed as parameter
    # Oterwise initialize base weights
    if args.load_checkpoint is not None :
        model.load_state_dict(torch.load(args.load_checkpoint, map_location=torch.device(DEVICE)))

    if args.optimizer_type == 'Adam' :
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.momentum, 0.999))
    elif args.optimizer_type == 'AdamW' :
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(args.momentum, 0.999))
    elif args.optimizer_type == 'SGD' :
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=(args.momentum), nesterov=False)
    else :
        raise ValueError("Unsupported optimizer type.\n   - available choices : ASG, Adam, AdamW")

    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss()

    return model, optimizer, criterion_slots, criterion_intents


'''
UTILITY
'''

def set_bert_input_and_slots (sample, lang, tokenizer):
    """
    Prepare BERT inputs and slot labels for training or evaluation.

    Args:
        sample (dict): Sample containing utterances, slots, and intents.
        lang (Lang): Language object for token mappings.
        tokenizer: BERT tokenizer.

    Returns:
        tuple: BERT input tensors, slot tensors, and special tokens.
    """
    # convert ids to words and tokenize them all at once
    sentences = [[lang.id2word[y] for y in x.int().tolist()] for x in sample['utterance']]
    sentences = [" ".join(x) for x in sentences]
    bert_input = tokenizer(sentences, return_tensors="pt", padding=True)
    bert_input = {k: v.to(DEVICE) for k, v in bert_input.items()}

    # generate bert_slots
    bert_slots = torch.ones((len(sentences), bert_input['input_ids'].shape[1]), dtype=torch.long, device=DEVICE) * PAD_TOKEN
    # keeps trak of the position of both sub-tokens and special tokens [CLS] and [SEP]
    special_tokens = {batch_id: set() for batch_id in range(len(sentences))}
    
    for batch_id, sentence in enumerate(sentences) :
        # [CLS] token need to be ignored on slot inference
        special_tokens[batch_id].add(0)
        
        counter = 1
        for pos, word in enumerate(sentence.split()) :
            subtokens = tokenizer(word, padding=False)["input_ids"][1:-1]
            curr_slot = sample['slots'][batch_id][pos]

            # assign to the first subtoken the slot of the word
            bert_slots[batch_id, counter] = curr_slot
            counter += 1

            remaining_sub_toks = len(subtokens) - 1
            if remaining_sub_toks == 0:
                continue
            
            special_tokens[batch_id].update(range(counter, counter + remaining_sub_toks)) 
            counter += remaining_sub_toks

        # also [SEP] token need to be ignored on slot inference
        special_tokens[batch_id].add(counter)
    
    # Transform to tensor and return dict.
    bert_slots = torch.Tensor(bert_slots).to(DEVICE)

    return bert_input, bert_slots, special_tokens

def get_subtok_poses (special_tokens, batch_size):
    """
    Get positions of sub-tokens for each batch.

    Args:
        special_tokens (dict): Dictionary of special token positions.
        batch_size (int): Size of the batch.

    Returns:
        dict: Subtoken positions for each batch.
    """
    subtoken_poses = copy.deepcopy(special_tokens)
    for batch_id in range(batch_size):
        subtoken_poses[batch_id].discard(0)                             # remove [CLS] token reference
        subtoken_poses[batch_id].discard(max(subtoken_poses[batch_id])) # remove [SEP] token reference

    return subtoken_poses


def save_model(model, filename='model.pt'):
    """
    Save the model's state dictionary.

    Args:
        model (torch.nn.Module): The model to save.
        filename (str, optional): Filename to save the model. Defaults to 'model.pt'.
    """
    path = os.path.join("bin", filename)
    torch.save(model.state_dict(), path)

def get_num_parameters(model):
    """
    Get the number of parameters in the model.

    Args:
        model (torch.nn.Module): The model to inspect.

    Returns:
        tuple: Total number of parameters and number of trainable parameters.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def get_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Model parameters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model args
    parser.add_argument("--bert_version", type=str, default="bert-base-uncased", required=False, help="bert version {bert-base-uncased, bert-base-cased}")
    parser.add_argument("--finetune_bert", default=False, action="store_true", required=False, help="if set BERT is finetuned, otherwise it's kept frozen")
    # component flags
    parser.add_argument("--merger_enable", default=False, action="store_true", required=False, help="enables merger component using self attention")
    parser.add_argument("--num_heads", type=int, default=1, required=False, help="number of attention heads")
    parser.add_argument("--dropout_enable", default=False, action="store_true", required=False, help="enables dropout")
    parser.add_argument("--int_dropout", type=float, default=0.1, required=False, help="intents dropout probability")
    parser.add_argument("--slot_dropout", type=float, default=0.1, required=False, help="slots dropout probability")

    # Training/Inference related args
    parser.add_argument("--lr", type=float, default=0.0001, required=False, help="learning rate")
    parser.add_argument("--n_epochs", type=int, default=200, required=False, help="number of epochs")
    parser.add_argument("--runs", type=int, default=5, required=False, help="number of runs trainings to make before plotting results")
    parser.add_argument("--train_bsize", type=int, default=128, required=False, help="training set batch size")
    parser.add_argument("--val_bsize", type=int, default=64, required=False, help="validation set batch size")
    parser.add_argument("--test_bsize", type=int, default=64, required=False, help="test set batch size")
    parser.add_argument("--optimizer_type", type=str, default="Adam", required=False, help="type of optimizer {SGD, Adam, AdamW}")
    parser.add_argument("--momentum", type=float, default=0.0, required=False, help="adds momentum to the defined optimizer")

    # Additional control flow arguments
    parser.add_argument("--load_checkpoint", type=str, default=None, required=False, help="path to weight checkpoint to load")
    parser.add_argument("--test_only",  default=False, action="store_true", required=False, help="avoid training the model. Perform inference on test set only")
    parser.add_argument("--save_model", default=False, action="store_true", required=False, help="store the model in 'model_bin/'")
    parser.add_argument("--enable_logger", default=False, action="store_true", required=False, help="log to wandb")

    return parser.parse_args()


def init_logger(args):
    """
    Initialize logging with Weights & Biases.

    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="NLU-part-2.2",

        # track hyperparameters and run metadata
        config={
        "learning_rate": args.lr,
        "dataset": "ATIS",
        "optimizer" : args.optimizer_type,
        "momentum" : args.momentum,
        "epochs": args.n_epochs,
        "dropout_enable": args.dropout_enable,
        "int_dropout": args.int_dropout if args.dropout_enable else 0,
        "slot_dropout": args.slot_dropout if args.dropout_enable else 0,
        "bert_version": args.bert_version,
        "fine_tune_bert": args.finetune_bert,
        "merger_enable": args.merger_enable,
        "n_attention_heads": args.num_heads if args.merger_enable else 0
        }

    )
