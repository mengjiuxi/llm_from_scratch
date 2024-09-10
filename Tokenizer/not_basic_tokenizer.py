"""
use regex split with GPT style
use special token
key implementation idea is to encode by group
in the basic tokenizer there is only one huge group
each group represents a split defined by the regex
note that text.encode("utf-8") returns bytes
to convert byte to 0...255, use list(bytes)
Note that in tiktoken, rank is only mapping from a sequences of bytes to a new idx 
which is equivalent to merge in this implementation
there is no vocab in tiktoken
[1, 2] -> 3
[1, 2, 3] -> 4
[3, 4] -> 5
to decoding, get the key, value of the "rank", no need for vocab
"""

from base import Tokenizer
import regex as re

ENDOFTEXT = "<|endoftext|>"
FIM_PREFIX = "<|fim_prefix|>"
FIM_MIDDLE = "<|fim_middle|>"
FIM_SUFFIX = "<|fim_suffix|>"
ENDOFPROMPT = "<|endofprompt|>"

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# cl100k_base in tiktoken
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer(Tokenizer):
    def __init__(self, pattern=None):
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if not pattern else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {
            ENDOFTEXT: 100257,
            FIM_PREFIX: 100258,
            FIM_MIDDLE: 100259,
            FIM_SUFFIX: 100260,
            ENDOFPROMPT: 100276
        }
        self.inverse_special_tokens = {
            100257: ENDOFTEXT,
            100258: FIM_PREFIX,
            100259: FIM_MIDDLE,
            100260: FIM_SUFFIX,
            100276: ENDOFPROMPT
        }

    def train(self, text, vocab_size):
        # create merges and vocab through chunk-wise merge
        if vocab_size < 256:
            raise ValueError("vocab size smaller than 256!")
        chunks = re.findall(self.compiled_pattern, text)
        # list of list [[chunks1, which is x, y, z, ...],[],[]]
        ids = [list(c.encode("utf-8")) for c in chunks]
        for i in range(vocab_size - 256):
            pair_freq = {}
            for chunk_ids in ids:
                pair_freq = self.get_pair_freq(chunk_ids, pair_freq)
            pair = max(pair_freq, key=pair_freq.get)
            idx = 256 + i
            ids = [self.merge(chunk_ids, pair, idx) for chunk_ids in ids]
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

    def _encode_chunk(self, text_bytes):
        """
        helper function to encode each chunk
        bytes are from text.encode("utf-8")
        """
        ids = list(text_bytes)
        while len(ids) >= 2:
            pair_freq = self.get_pair_freq(ids)
            pair = min(pair_freq, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = self.merge(ids, pair, self.merges[pair])
        return ids
    
    def encode_ordinary(self, text):
        """
        encode function without special tokens
        operating on each chunk, return a single sequence of integer
        """
        text_chunks = re.findall(self.compiled_pattern, text)
        ids = []
        for chunk in text_chunks:
            chunk_ids = self._encode_chunk(chunk.encode("utf-8"))
            ids.extend(chunk_ids)
        return ids
    
    def encode(self, text, allowed_special="none_raise"):
        """
        allowed_special can be all|none|none_raise or custom special tokens
        same as in tiktoken
        """
        special = None
        if allowed_special == "all":
            # use all defined special tokens
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            # special token should not appear in text, if so raise error
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            # if input is custom set of special tokens
            # then only enable those that are in the defined special tokens
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allow_special={allowed_special} not found!")
        if not special:
            # print("No special token used.")
            return self.encode_ordinary(text)
        # surround pattern with () to keep the separator after re.split(), 
        # separator aka the special token
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
        # example special_chunks
        # [text, text, |special token|, text,  |s_token| ...]

        ids = []
        for chunk in special_chunks:
            if chunk in special:
                ids.append(special[chunk])
            else:
                ids.extend(self.encode_ordinary(chunk))
        return ids

    def decode(self, ids):
        byte_list = []
        for i in ids:
            if i in self.vocab:
                byte_list.append(self.vocab[i])
            elif i in self.inverse_special_tokens:
                byte_list.append(self.inverse_special_tokens[i].encode("utf-8"))
            else:
                raise ValueError(f"Invalid token: {i}!")
        text = b"".join(byte_list).decode("utf-8", errors="replace")
        return text
    

# text = "Mock Interview for Data Scientist Part 1: Problem Solving (45 minutes) Interviewer: Let’s start with a problem that we often encounter when working with clients. Imagine you’re working with a customer from the energy sector who wants to reduce their operational costs by optimizing their energy consumption. They have a large dataset of energy usage at different times of the day, spanning multiple years. They also have external data such as weather conditions, operational schedules, and energy prices. 	1.	Question: How would you approach solving this problem? Follow-up: How would you engage with the customer to ensure. <|endoftext|>" 
# tk = RegexTokenizer()
# tk.train(text, 500)
# print(tk.decode(tk.encode("hello world!!!? (안녕하세요!) lol123 😉,<|endoftext|>", allowed_special="none")))

# import tiktoken
# enc = tiktoken.get_encoding("cl100k_base") # this is the GPT-4 tokenizer
# ids = enc.encode("hello world!!!? (안녕하세요!) lol123 😉")
# text = enc.decode(ids) # get the same text back
# print(text)