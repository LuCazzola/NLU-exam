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


def train_step(model, data_loader, optimizer, criterion_slots, criterion_intents, lang, clip=5):
    model.train()
    loss_array = []
    for sample in data_loader:
        optimizer.zero_grad() # Zeroing the gradient

        bert_input, bert_slots = set_bert_input_and_slots (sample, lang, model.tokenizer)
        slots, intents = model(bert_input)

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
    loss_array = []
    ref_intents = []
    hyp_intents = []
    ref_slots = []
    hyp_slots = []

    model.eval()
    #softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data_loader:
            bert_input, bert_slots = set_bert_input_and_slots (sample, lang, model.tokenizer)
            slots, intents = model(bert_input)
            
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
            print("SLOTS SHAPE", slots.shape)
            print("SLOTS", slots)
            print("OUTPUT SLOTS SHAPE", output_slots.shape)
            print("OUTPUT SLOTS", output_slots)
            exit()
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
    losses_train = []
    losses_val = []
    sampled_epochs = []
    best_model = None
    best_f1 = 0
    for epoch in tqdm(range(1,n_epochs+1)):
        loss = train_step(model, train_loader, optimizer, criterion_slots, criterion_intents, lang, clip=clip)

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

def init_components (args, lang) :

    model = ModelIAS(len(lang.slot2id), len(lang.intent2id), len(lang.word2id),
        dropout_enable=args.dropout_enable,
        emb_dropout=args.emb_dropout,
        out_dropout=args.out_dropout,
        bert_version=args.bert_version,
        num_heads=args.num_heads
    ).to(DEVICE)

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
    # convert ids to words and tokenize them all at once
    sentences = [[lang.id2word[y] for y in x.int().tolist()] for x in sample['utterance']]
    sentences = [" ".join(x) for x in sentences]
    bert_input = tokenizer(sentences, return_tensors="pt", padding=True)
    bert_input = {k: v.to(DEVICE) for k, v in bert_input.items()}

    # generate bert_slots
    bert_slots = torch.ones((len(sentences), bert_input['input_ids'].shape[1]), dtype=torch.long, device=DEVICE) * PAD_TOKEN
    for batch_id, sentence in enumerate(sentences) :
        
        counter = 1
        for pos, word in enumerate(sentence.split()) :
            tokens = tokenizer(word, padding=False)["input_ids"][1:-1]
            curr_slot = sample['slots'][batch_id][pos]

            for subtok in tokenizer.convert_ids_to_tokens(tokens):
                if not str(subtok).startswith("##") :
                    # subtokens will be skipped
                    bert_slots[batch_id, counter] = curr_slot
                counter += 1
    
    # Transform to tensor and return dict.
    bert_slots = torch.Tensor(bert_slots).to(DEVICE)

    return bert_input, bert_slots


def save_model(model, filename='model.pt'):
    path = os.path.join("bin", model_name)
    torch.save(model.state_dict(), path)

def get_num_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def get_args():
    parser = argparse.ArgumentParser(
        description="Model parameters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model args
    parser.add_argument("--bert_version", type=str, default="bert-base-uncased", required=False, help="bert version {bert-base-uncased, bert-base-cased}")
    parser.add_argument("--num_heads", type=int, default=1, required=False, help="number of attention heads")
    parser.add_argument("--dropout_enable", default=False, action="store_true", required=False, help="enables dropout")
    parser.add_argument("--emb_dropout", type=float, default=0.1, required=False, help="embedding layer dropout probability (requires dropout enabled to function)")
    parser.add_argument("--out_dropout", type=float, default=0.1, required=False, help="rnn output layer dropout probability (requires dropout enabled to function)")

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
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="NLU-part2",

        # track hyperparameters and run metadata
        config={
        "learning_rate": args.lr,
        "dataset": "ATIS",
        "optimizer" : args.optimizer_type,
        "momentum" : args.momentum,
        "epochs": args.n_epochs,
        }
    )