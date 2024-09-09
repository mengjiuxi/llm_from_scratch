from base import Tokenizer
class BasicTokenizer(Tokenizer):
    """
    vocab and merges are obtained in training phase
    encode and decodes works with the trained merges and vocab
    """
    def __init__(self):
        super().__init__()
        # base class already define vocab and merges, no need to overwrite
        # self.vocab = {idx: bytes([idx]) for idx in range(256)} # int : bytes 
        # self.merges = {} # pair : symbol

    # move to base
    # # get the freq count of each pair in the string
    # def get_pair_freq(self, ids):
    #     counts = {}
    #     for pair in zip(ids, ids[1:]):
    #         counts[pair] = counts.get(pair, 0) + 1
    #     return counts

    # # merge pair in the string with new idx
    # def merge(self, ids, pair, idx):
    #     new_ids = [] # list of ints after merge
    #     i = 0
    #     while i < len(ids):
    #         if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
    #             new_ids.append(idx)
    #             i += 2
    #         else:
    #             new_ids.append(ids[i])
    #             i += 1
    #     return new_ids

    def train(self, text, vocab_size):
        assert vocab_size >= 256
        # generate vocab and merges
        ids = list(text.encode("utf-8"))
        num_merges = vocab_size - 256
        for i in range(num_merges):
            pair_freq = self.get_pair_freq(ids)
            # get the pair with the maximum freq to merge first
            pair = max(pair_freq, key=pair_freq.get)
            idx = 256 + i
            # print(f"merging {pair} into a new token {idx}")
            ids = self.merge(ids, pair, idx)
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
        
    def encode(self, text):
        # a list of bytes
        ids = list(text.encode("utf-8"))
        while (len(ids) >= 2):
            freq_count = self.get_pair_freq(ids)
            # start with pair with smallest merge idx
            # since >= python 3.7, dictionary maintains insertion order
            # we are essentially starting from pairs that had been merged first
            pair = min(freq_count, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = self.merge(ids, pair, idx)
        return ids

    def decode(self, ids):
        # ids is a list of bytes 
        tokens = b"".join(self.vocab[idx] for idx in ids)
        # convert byte string back to human readable texts
        text = tokens.decode("utf-8", errors="replace")
        return text
    

# text = "Mock Interview for Data Scientist Part 1: Problem Solving (45 minutes) Interviewer: Let’s start with a problem that we often encounter when working with clients. Imagine you’re working with a customer from the energy sector who wants to reduce their operational costs by optimizing their energy consumption. They have a large dataset of energy usage at different times of the day, spanning multiple years. They also have external data such as weather conditions, operational schedules, and energy prices. 	1.	Question: How would you approach solving this problem? Follow-up: How would you engage with the customer to ensure" 
# tk = BasicTokenizer()
# tk.train(text, 500)
# print(tk.decode(tk.encode("hello, my world!")))