"""
Module for data loading and preprocessing.
"""
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from functools import partial

# Define the device on which to run models
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
CLASSES
"""

class Lang():
    """
    A class to handle vocabulary and word-to-ID mappings.

    Args:
        corpus (list of str): List of sentences where each sentence is a string of space-separated tokens.
        special_tokens (list of str, optional): List of special tokens to include in the vocabulary. Default is an empty list.
    """
    def __init__(self, corpus, special_tokens=[]):
        self.word2id = self.get_vocab(corpus, special_tokens)
        self.id2word = {v:k for k, v in self.word2id.items()}
    
    def get_vocab(self, corpus, special_tokens=[]):
        """
        Generate vocabulary mapping from tokens to IDs.

        Args:
            corpus (list of str): List of sentences where each sentence is a string of space-separated tokens.
            special_tokens (list of str, optional): List of special tokens to include in the vocabulary. Default is an empty list.

        Returns:
            dict: A dictionary mapping tokens to unique IDs.
        """
        output = {}
        i = 0 
        for st in special_tokens:
            output[st] = i
            i += 1
        for sentence in corpus:
            for w in sentence.split():
                if w not in output:
                    output[w] = i
                    i += 1
        return output


class PennTreeBank (data.Dataset):
    """
    A Dataset class for the PennTreeBank corpus.

    Args:
        corpus (list of str): List of sentences where each sentence is a string of space-separated tokens.
        lang (Lang): An instance of the Lang class containing word-to-ID mappings.
    """
    def __init__(self, corpus, lang):
        self.source = []
        self.target = []
        
        for sentence in corpus:
            self.source.append(sentence.split()[0:-1]) # We get from the first token till the second-last token
            self.target.append(sentence.split()[1:]) # We get from the second token till the last token
            # See example in section 6.2
        
        self.source_ids = self.mapping_seq(self.source, lang)
        self.target_ids = self.mapping_seq(self.target, lang)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        src= torch.LongTensor(self.source_ids[idx])
        trg = torch.LongTensor(self.target_ids[idx])
        sample = {'source': src, 'target': trg}
        return sample
    
    def mapping_seq(self, data, lang): # Map sequences of tokens to corresponding computed in Lang class
        """
        Map sequences of tokens to corresponding IDs.

        Args:
            data (list of list of str): List of sequences, where each sequence is a list of tokens.
            lang (Lang): An instance of the Lang class containing word-to-ID mappings.

        Returns:
            list of list of int: List of sequences with tokens replaced by their corresponding IDs.
        """
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq:
                if x in lang.word2id:
                    tmp_seq.append(lang.word2id[x])
                else:
                    print('OOV found!')
                    print('You have to deal with that') # PennTreeBank doesn't have OOV but "Trust is good, control is better!"
                    break
            res.append(tmp_seq)
        return res


"""
UTILITY FUNCTIONS
"""

def read_file(path, eos_token="<eos>"):
    """
    Read a text file and append an end-of-sequence token to each line.

    Args:
        path (str): Path to the text file.
        eos_token (str, optional): End-of-sequence token to append. Default is "<eos>".

    Returns:
        list of str: List of sentences with each line ending in the end-of-sequence token.
    """
    output = []
    with open(path, "r") as f:
        for line in f.readlines():
            output.append(line.strip() + " " + eos_token)
    return output


def collate_fn(data, pad_token):
    """
    Custom collate function for padding and batching data.

    Args:
        data (list of dict): List of samples where each sample is a dictionary with 'source' and 'target' tensors.
        pad_token (int): Token ID used for padding.

    Returns:
        dict: A dictionary containing padded 'source' and 'target' tensors and total number of tokens.
    """
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape 
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(pad_token)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq # We copy each sequence into the matrix
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths
    
    # Sort data by seq lengths
    data.sort(key=lambda x: len(x["source"]), reverse=True) 
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]

    source, _ = merge(new_item["source"])
    target, lengths = merge(new_item["target"])
    
    new_item["source"] = source.to(DEVICE)
    new_item["target"] = target.to(DEVICE)
    new_item["number_tokens"] = sum(lengths)
    return new_item


def init_data(args):
    """
    Initialize data loaders and vocabulary for training, validation, and test datasets.

    Args:
        args (argparse.Namespace): Arguments containing batch sizes for training, validation, and test datasets.

    Returns:
        Lang: An instance of the Lang class containing word-to-ID mappings.
        DataLoader: DataLoader for training dataset.
        DataLoader: DataLoader for validation dataset.
        DataLoader: DataLoader for test dataset.
    """
    # Read raw corpus
    train_raw = read_file("dataset/PennTreeBank/ptb.train.txt")
    dev_raw = read_file("dataset/PennTreeBank/ptb.valid.txt")
    test_raw = read_file("dataset/PennTreeBank/ptb.test.txt")

    # Define vocabulary (only on training)
    lang = Lang(train_raw, ["<pad>", "<eos>"])

    # Define dataset classes
    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)

    # Define assiciated dataloaders {Training, Validation, Test}
    train_loader = DataLoader(train_dataset, batch_size=args.train_bsize, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),  shuffle=True)
    val_loader = DataLoader(dev_dataset, batch_size=args.val_bsize, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    test_loader = DataLoader(test_dataset, batch_size=args.test_bsize, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))

    return lang, train_loader, val_loader, test_loader