class Tokenizer:
    """
    Base class for all Tokenizers
    including save and load model function
    """
    def __init__(self):
        self.merges = {} # (int, int) -> int
        self.pattern = "" # str
        self.special_tokens = {} # str -> int
        self.vocab = self._build_vocab() # int -> bytes
    
    # In user defined base classes, abstract methods should raise this exception when they require derived classes to override the method, or while the class is being developed to indicate that the real implementation still needs to be added.
    def train(self, text, vocab_size):
        raise NotImplementedError
    
    def encode(self, text):
        raise NotImplementedError
    
    def decode(self, ids):
        raise NotImplementedError
    
    # get the freq count of each pair in the string
    def get_pair_freq(self, ids, legacy=None):
        counts = {} if not legacy else legacy
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    # merge pair in the string with new idx
    def merge(self, ids, pair, idx):
        new_ids = [] # list of ints after merge
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def _build_vocab(self):
        # bytes([list]) convert iterable list to bytes
        # whereas bytes(num) returns num empty bytes
        vocab = {idx : bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            # bytes + bytes is concatenation
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab
    

    def save(self, file_prefix):
        """save merges and special tokens to a file"""
        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            f.write(f"{self.pattern}\n")

            # store the number of special tokens
            f.write(f"{len(self.special_tokens)}\n")
            # store the str:int pair
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            # store pairs from merges 
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")
    
    def load(self, model_file):
        assert model_file.endswith(".model")
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, 'r', encoding="utf-8") as f:
            self.pattern = f.readline().strip()
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            
            # the remaining lines are merge pairs
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1

        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()