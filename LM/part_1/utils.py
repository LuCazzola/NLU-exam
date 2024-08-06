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
    '''
    Computes and stores the vocabulary, mapping:
    - Word -> ids
    - ids -> word
    
    Parameters:
        corpus (list of str): List of sentences forming the corpus.
        special_tokens (list of str, optional): List of special tokens to include in the vocabulary. Default is [].
    '''
    def __init__(self, corpus, special_tokens=[]):
        self.word2id = self.get_vocab(corpus, special_tokens)
        self.id2word = {v:k for k, v in self.word2id.items()}
    
    def get_vocab(self, corpus, special_tokens=[]):
        '''
        Constructs the vocabulary from the given corpus and special tokens.
        
        Parameters:
            corpus (list of str): List of sentences forming the corpus.
            special_tokens (list of str): List of special tokens to include in the vocabulary.
        
        Returns:
            dict: Mapping of words to unique ids.
        '''
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
    '''
    Custom Dataset for Penn Tree Bank corpus.
    
    Parameters:
        corpus (list of str): List of sentences forming the corpus.
        lang (Lang): Language object containing the vocabulary mappings.
    '''
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
        '''
        Maps sequences of tokens to their corresponding ids using the Lang class.
        
        Parameters:
            data (list of list of str): Sequences of tokens.
            lang (Lang): Language object containing the vocabulary mappings.
        
        Returns:
            list of list of int: Mapped sequences of token ids.
        '''

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
    '''
    Loads the given corpus from a file and appends an end-of-sentence token to each sentence.
    
    Parameters:
        path (str): Path to the corpus file.
        eos_token (str): End-of-sentence token to be appended. Default is "<eos>".
    
    Returns:
        list of str: List of sentences from the corpus.
    '''
    output = []
    with open(path, "r") as f:
        for line in f.readlines():
            output.append(line.strip() + " " + eos_token)
    return output


def collate_fn(data, pad_token):
    '''
    Custom collate function to merge sequences and pad them to the same length.
    
    Parameters:
        data (list of dict): Batch of data samples.
        pad_token (int): Token used for padding sequences.
    
    Returns:
        dict: Batch with padded sequences and additional metadata.
    '''

    def merge(sequences):
        '''
        Merges a list of sequences into a single tensor, padding to the maximum length.
        
        Parameters:
            sequences (list of torch.Tensor): List of sequences to merge.
        
        Returns:
            tuple: Padded tensor of sequences and list of original sequence lengths.
        '''
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
    '''
    Instantiates data sources and DataLoader objects.
    
    Parameters:
        args (argparse.Namespace): Arguments containing batch sizes for train, val, and test sets.
    
    Returns:
        tuple: Language object, train DataLoader, val DataLoader, test DataLoader.
    '''
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