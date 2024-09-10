import tiktoken
from not_basic_tokenizer import RegexTokenizer

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
GPT4_SPECIAL_TOKENS = {
    '<|endoftext|>': 100257,
    '<|fim_prefix|>': 100258,
    '<|fim_middle|>': 100259,
    '<|fim_suffix|>': 100260,
    '<|endofprompt|>': 100276
}

def bpe(mergeable_ranks, token, max_rank):
    """
    """
    # convert each char in token to bytes
    # find the pair with the minimum rank, mark its index, then merge this pair 
    parts = [bytes([b]) for b in token]
    while True:
        min_idx = None
        min_rank = None
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank
        # if min_rank is None: no matching pair found to merge
        # (max_rank is not None and min_rank >= max_rank): 
        # use max_rank to limit the number of merges
        # it is used to get to the rank - 1 merge
        # that is the last step before merge into "token"
        # we are guaranteed to have two parts
        if min_rank is None or (max_rank is not None and min_rank >= max_rank):
            break
        assert min_idx is not None
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]
    
    return parts
                
def recover_merges(mergeable_ranks):
    """
    convert GPT_style mergeable_ranks to merges
    honestly, I think mergeable ranks is cleaner
    """
    merges = {}
    for token, rank in mergeable_ranks.items():
        if len(token) == 1:
            continue
        pair = tuple(bpe(mergeable_ranks, token, max_rank=rank))
        assert len(pair) == 2
        # get the rank of each part
        ix0 = mergeable_ranks[pair[0]]
        ix1 = mergeable_ranks[pair[1]]
        merges[(ix0, ix1)] = rank
    return merges

class GPT4Tokenizer(RegexTokenizer):
    def __init__(self):
        super().__init__(pattern=GPT4_SPLIT_PATTERN)
        enc = tiktoken.get_encoding("cl100k_base")
        mergeable_ranks = enc._mergeable_ranks
        self.merges = recover_merges(mergeable_ranks)
        vocab = {idx : bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] =  vocab[p0] + vocab[p1]
        self.vocab = vocab
    
        # permute for what reason?
        self.byte_shuffle = {i: mergeable_ranks[bytes([i])] for i in range(256)}
        self.inverse_byte_shuffle = {v: k for k, v in self.byte_shuffle.items()}
        
    def _encode_chunk(self, text_bytes):
        text_bytes = bytes(self.byte_shuffle[b] for b in text_bytes)
        ids = super()._encode_chunk(text_bytes)
        return ids

    def decode(self, ids):
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text_bytes = bytes(self.inverse_byte_shuffle[b] for b in text_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text
    
    def train(self, text, vocab_size):
        raise NotImplementedError("Pretrained model, cannot be saved.")
    
    def load(self, model_file):
        raise NotImplementedError("Pretrained mode, what are you thinking?")

 
# tk = GPT4Tokenizer()
# print(tk.decode(tk.encode("hello world!!!? (안녕하세요!) lol123 😉,<|endoftext|>", allowed_special="none")))
