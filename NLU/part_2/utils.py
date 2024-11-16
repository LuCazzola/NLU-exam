import os
import json
from pprint import pprint
import random
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAD_TOKEN = 0

"""
CLASSES
"""

class Lang():
    """
    Class to handle language-related mappings for words, slots, and intents.

    Args:
        words (list of str): List of words in the corpus.
        intents (list of str): List of intent labels.
        slots (list of str): List of slot labels.
        cutoff (int, optional): Minimum frequency of words to be included in vocabulary. Defaults to 0.
    """
    def __init__(self, words, intents, slots, cutoff=0):
        self.word2id = self.w2id(words, cutoff=cutoff, unk=True)
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2word = {v:k for k, v in self.word2id.items()}
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}
        
    def w2id(self, elements, cutoff=None, unk=True):
        vocab = {'pad': PAD_TOKEN}
        if unk:
            vocab['unk'] = len(vocab)
        count = Counter(elements)
        for k, v in count.items():
            if v > cutoff:
                vocab[k] = len(vocab)
        return vocab
    
    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab['pad'] = PAD_TOKEN
        for elem in elements:
            vocab[elem] = len(vocab)
        return vocab


class IntentsAndSlots (data.Dataset):
    """
    Dataset class for handling intent and slot classification tasks.

    Args:
        dataset (list of dict): List of samples, where each sample is a dictionary containing 'utterance', 'slots', and 'intent'.
        lang (Lang): Language object for mappings.
        unk (str, optional): Token for unknown words. Defaults to 'unk'.
    """
    def __init__(self, dataset, lang, unk='unk'):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.unk = unk
        
        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])

        self.utt_ids = self.mapping_seq(self.utterances, lang.word2id)
        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt = torch.Tensor(self.utt_ids[idx])
        slots = torch.Tensor(self.slot_ids[idx])
        intent = self.intent_ids[idx]
        sample = {'utterance': utt, 'slots': slots, 'intent': intent}
        return sample
    
    # Auxiliary methods
    
    def mapping_lab(self, data, mapper):
        """
        Map labels to indices.

        Args:
            data (list of str): List of labels.
            mapper (dict): Dictionary mapping labels to indices.

        Returns:
            list of int: List of indices corresponding to the labels.
        """
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]
    
    def mapping_seq(self, data, mapper): # Map sequences to number
        """
        Map sequences of words to indices.

        Args:
            data (list of str): List of sequences, where each sequence is a string of words.
            mapper (dict): Dictionary mapping words to indices.

        Returns:
            list of list of int: List of sequences, where each sequence is a list of indices.
        """
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq.split():
                if x in mapper:
                    tmp_seq.append(mapper[x])
                else:
                    tmp_seq.append(mapper[self.unk])
            res.append(tmp_seq)
        return res


"""
UTILITY
"""

def load_data(path):
    """
    Load data from a JSON file.

    Args:
        path (str): Path to the JSON file.

    Returns:
        list of dict: List of samples loaded from the JSON file.
    """
    dataset = []
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset

def collate_fn(data):
    """
    Custom collate function to merge sequences and pad them.

    Args:
        data (list of dict): List of samples, where each sample is a dictionary containing 'utterance', 'slots', and 'intent'.

    Returns:
        dict: Dictionary containing padded 'utterances', 'y_slots', 'intents', and 'slots_len'.
    """
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len 
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape 
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(PAD_TOKEN)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq # We copy each sequence into the matrix
        # print(padded_seqs)
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths
    # Sort data by seq lengths
    data.sort(key=lambda x: len(x['utterance']), reverse=True) 
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]
        
    # We just need one length for packed pad seq, since len(utt) == len(slots)
    src_utt, _ = merge(new_item['utterance'])
    y_slots, y_lengths = merge(new_item["slots"])
    intent = torch.LongTensor(new_item["intent"])
    
    src_utt = src_utt.to(DEVICE)
    y_slots = y_slots.to(DEVICE)
    intent = intent.to(DEVICE)
    y_lengths = torch.LongTensor(y_lengths).to(DEVICE)
    
    new_item["utterances"] = src_utt
    new_item["intents"] = intent
    new_item["y_slots"] = y_slots
    new_item["slots_len"] = y_lengths
    return new_item


"""
INITIALIZERS
"""

def init_data(args):
    """
    Initialize data loaders and language mappings.

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing batch sizes.

    Returns:
        tuple: A tuple containing:
            - Lang: Language object with mappings.
            - DataLoader: DataLoader for training data.
            - DataLoader: DataLoader for validation data.
            - DataLoader: DataLoader for test data.
    """
    tmp_train_raw = load_data(os.path.join('dataset','ATIS','train.json'))
    portion = 0.10

    intents = [x['intent'] for x in tmp_train_raw] # We stratify on intents
    count_y = Counter(intents)

    labels = []
    inputs = []
    mini_train = []

    for id_y, y in enumerate(intents):
        if count_y[y] > 1: # If some intents occurs only once, we put them in training
            inputs.append(tmp_train_raw[id_y])
            labels.append(y)
        else:
            mini_train.append(tmp_train_raw[id_y])
    # Random Stratify
    X_train, X_val, y_train, y_val = train_test_split(inputs, labels, test_size=portion, random_state=42, shuffle=True, stratify=labels)
    X_train.extend(mini_train)
    train_raw = X_train
    val_raw = X_val

    test_raw = load_data(os.path.join('dataset','ATIS','test.json'))
    y_test = [x['intent'] for x in test_raw]

    words = sum([x['utterance'].split() for x in train_raw], []) # No set() since we want to compute # the cutoff
    corpus = train_raw + val_raw + test_raw # We do not wat unk labels, however this depends on the research purpose
    slots = sorted(list(set(sum([line['slots'].split() for line in corpus], []))))
    intents = sorted(list(set([line['intent'] for line in corpus])))
    lang = Lang(words, intents, slots, cutoff=0)

    # Create our datasets
    train_dataset = IntentsAndSlots(train_raw, lang)
    val_dataset = IntentsAndSlots(val_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)

    # Dataloader instantiations
    train_loader = DataLoader(train_dataset, batch_size=args.train_bsize, collate_fn=collate_fn,  shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.val_bsize, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.test_bsize, collate_fn=collate_fn)

    return lang, train_loader, val_loader, test_loader
