from conllu import parse
import re
import torch

def read_cupt(path):
    """
    Read CoNLL-U Plus (CUPT) formatted file and parse it into a list of sentences.

    Parameters:
    - path (str): The path to the CoNLL-U Plus formatted file.

    Returns:
    - list: A list of sentences, where each sentence is represented as a list of token dictionaries.
    
    Example:
    --------
    >>> sentences = read_cupt('path/to/your/file.cupt')
    >>> first_sentence = sentences[0]
    >>> second_token = first_sentence[1]
    >>> lemma = second_token['lemma'] (available keys : 'id', 'form', 'lemma', 'upos', 'xpos', 'feats', 'head', 'deprel', 'deps', 'misc', 'parseme:mwe')
    """
    with open(path, 'r', encoding='utf-8') as file:
        data = file.read()
        sentences = parse(data)
    
    return sentences

def extract_numbers(sentence):
    # Use regular expression to find all numbers in the sentence
    numbers = re.findall(r'\d+', sentence)
    return numbers

class EarlyStopping:
    def __init__(self, patience=5, delta=0.00001, path='best_score.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 5
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        '''Saves model when validation loss decreases.'''
        torch.save(model.state_dict(), self.path)