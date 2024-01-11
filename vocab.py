import pandas as pd
import ast 

class Vocab:
    def __init__(self, csv_file):
        self.word_to_index = {}
        self.index_to_word = {}
        self.df = pd.read_csv(csv_file, sep="\t", 
                              converters={"token_list": ast.literal_eval, 
                                          "lemmas": ast.literal_eval, 
                                          'labels': ast.literal_eval})
        self.build_vocab()

    def build_vocab(self):
        for index, row in self.df.iterrows():
            token_list = row['token_list']
            for word in token_list:
                if word not in self.word_to_index:
                    index = len(self.word_to_index)
                    self.word_to_index[word] = index
                    self.index_to_word[index] = word

    def get_word_index(self, word):
        return self.word_to_index.get(word)

    def get_index_word(self, index):
        return self.index_to_word.get(index)

if __name__ == '__main__':
    vocab = Vocab('train_BIGO.csv')
    print(vocab.get_word_index('Paget'))