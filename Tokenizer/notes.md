
# mergeable_ranks {} : token_bytes, token

# How does tiktoken merge ?

# [12, 42, 52, 516, 62, 23, 63, 22]

# find the the minimum rank first, mark its position
# concatenate the pair 
# [12, [42, 52], 516, 62, 23, 63, 22]

# do it again

# [12, [[42, 52], 516], 62, 23, 63, 22]
# it is possible for the part list to look like this before replacing with another token
# so token_bytes can contain one to multiple bytes, and that corresponds to one new token


self._decoder = {token: token_bytes for token_bytes, token in mergeable_ranks.items()}

# summary on tiktoken _ducational.py and comparison with RegexTokenizer
1. self._decoder reverse the mergeable_ranks
2. encode calls bpe_encode() and returns a list of tokens
3. bpe_encode(mergeable_ranks, input)
4. decode_bytes() return b"".join(self._decoder[token] for token in tokens)
5. decode() return self.decode_bytes(tokens).decode("utf-8", errors="replace")
6. decode_tokens_bytes() return [self._decoder[token] for token in tokens] 
7. train() creates the mergeable_ranks by calling bpe_train()
8. bpe_encode() takes the mergeable_rank and encode the input


bpe_encode()
convert multiple bytes into lists of single bytes 
find pair with minimum rank
then merge pair in the list, so they become a single element
keep doing this until no more encodeable pair can be found
finally replace encoded individual element with their rank and return the encoded 

inside bpe_train()
words are first transformed to list of list of bytes

### what does ranks look like?

ranks is initalised with 0-255 bytes, each bytes corresponds to a rank 0 - 255
for each most_frequent_pair, we add ranks[pair[0] + pair[1]] = token to the ranks
where + is the concatenation of two bytes object 
after the first round
(b'0', b'1') : 256 is the most common pair
so ranks[b'01'] = 256, which gives
[b'0: 0, b'1: 1, b'2: 2, ... b'255: 255, b'01':256,]

then we update words by merging all the (0, 1) pairs
for example you have [b'h', b'e', b'l', b'l', b'o'] and (b'l', b'o') is the most common pair
then after merge you will get [b'h', b'e', b'l', b'lo'] where b'lo' is a single element 

in summary, each merge is on two elements
each element can be of multiple bytes, concatenated together, e.g. b'fdsafs'
mergeable_ranks is just a mapping between a bunch of bytes to a rank

### RegexTokenizer implementation

what does vocab look like again?
each element is expressed by the base 0 - 255 bytes 
vocab[idx] = vocab[p0] + vocab[p1]
vocab is the same as mergeable_ranks
[b'0': 0, b'1': 0, b'2': 0, ..., (b'01'): , (b'24'), (b'39'), (b'0124'), (b'2439'), (b'012439'), ...]

what does merges look like?
in the merge function, each pair is replaced by the rank (a new byte)
in the train, for each iteration
each pair is replaced by a new idx, which is a single byte
so the merges is (single byte, single byte) -> rank

### difference between two implementation?

merges is an intermediate container only for better understand how the pair is merged
it is not necessary
both implementation decode from the same type of dictionary,
that is mergeable_ranks for GPT4 and vocab for Regex