from conllu import parse
import re

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