import spacy
import os
import config, pickle
# import en_core_web_sm

# spacy_eng = en_core_web_sm.load()
spacy_eng = spacy.load("en_core_web_sm")
class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<start>", 2: "<end>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<start>": 1, "<end>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        
        if not os.path.isfile(config.VOCAB_FILE):
            print("Building vocabulary")
        # frequencies = {}
            idx = 4
            for sentence in sentence_list:
                for word in self.tokenizer_eng(sentence):
                    if word not in self.stoi:

                        self.stoi[word] = idx
                        self.itos[idx] = word
                        idx += 1
            with open(config.VOCAB_FILE, "wb") as f:
                pickle.dump(self.stoi, f)
        else:
            print("Vocab used out of dir....")
            with open(config.VOCAB_FILE, "rb") as f:
                self.stoi = pickle.load(f)
        return self.stoi
    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]