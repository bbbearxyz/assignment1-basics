import os
from typing import BinaryIO
import regex as re
from collections import defaultdict
import time
import json
from typing import Iterator, Iterable
import multiprocessing as mp
from functools import lru_cache
import numpy as np


DEBUG = True
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def str_to_tuple_bytes(s: str) -> tuple[bytes]:
    return tuple(bytes([c]) for c in s.encode("utf-8"))

def pre_tokenization(
    input_path: str,
    start: int,
    end: int,
    special_tokens: list[str] 
):
    count : dict[tuple[bytes], int] = defaultdict(int)
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        parts = re.split("|".join(map(re.escape, special_tokens)), chunk)

        for part in parts:
            for match in re.finditer(PAT, part):
                str = match.group()
                if str in special_tokens:
                    continue
                matchBytes = str_to_tuple_bytes(str)
                count[matchBytes] += 1
    return count

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    ## Usage
    vocab : dict[int, bytes] = {}
    merges : list[tuple[bytes, bytes]] = []
    count : dict[tuple[bytes], int] = defaultdict(int)
    num_processes = 10
    tasks : list[tuple[int, int]]


    start_time = time.time()
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        num_processes = min(num_processes, len(boundaries) - 1)
        tasks = [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]

    with mp.Pool(num_processes) as pool:
        results = pool.starmap(pre_tokenization, tasks)
        for c in results:
            for key, value in c.items():
                count[key] += value

    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    for i in range(256):
        vocab[i + len(special_tokens)] = bytes([i])

    test : dict[tuple[bytes, bytes], int] = defaultdict(int)
    # test记录多个token
    for match in count:
        for idx in range(len(match) - 1):
            test[(match[idx], match[idx + 1])] += count[match]

    # heap = []
    # for key, value in test.items():
    #     heapq.heappush(heap, (-value, key))

    mid_time = time.time()
    time_find_max = 0
    while len(vocab) < vocab_size:
        # merge
        first = time.time()

        pair = max(test.items(), key=lambda kv: (kv[1], kv[0]))[0]
        # 懒删除
        # while heap:
        #     freq, pair = heapq.heappop(heap)
        #     freq = -freq
        #     if test[pair] == freq and freq > 0:
        #         break


        time_find_max += (time.time() - first)
        new_token = pair[0] + pair[1]

        for match in list(count.keys()):
            change = False
            result = []
            for idx in range(len(match) - 1):
                # 只更新命中的
                if match[idx] == pair[0] and match[idx + 1] == pair[1]:
                    change = True
                    break
            if change:
                idx = 0
                while idx < len(match):
                    if idx < (len(match) - 1) and match[idx] == pair[0] and match[idx + 1] == pair[1]:
                        result.append(new_token)
                        idx += 2
                    else:
                        result.append(match[idx])
                        idx += 1

                # 更新新的match和result
                # 优化: 理论上只需要更新修改后的heap
                for idx in range(len(match) - 1):
                    test[(match[idx], match[idx + 1])] -= count[match]
                    # heapq.heappush(heap, (-test[(match[idx], match[idx + 1])], (match[idx], match[idx + 1])))

                for idx in range(len(result) - 1):
                    test[(result[idx], result[idx + 1])] += count[match]
                    # heapq.heappush(heap, (-test[(result[idx], result[idx + 1])], (result[idx], result[idx + 1])))
                count[tuple(result)] += count.pop(match)
        merges.append(pair)
        vocab[len(vocab)] = new_token
    
    end_time = time.time()
    print("pre token cost", mid_time - start_time)
    print("merge cost", end_time - mid_time)
    print("find max", time_find_max)
    return (vocab, merges)

@lru_cache
def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation. This function is taken
    from the GPT-2 code.

    For example, `chr(0)` is `\x00`, which is an unprintable character:

    >>> chr(0)
    '\x00'
    >>> print(chr(0))

    As a result, this function returns a dictionary `d` where `d[0]` returns `Ā`.
    The bytes that are visually printable keep their original string representation [1].
    For example, `chr(33)` returns `!`, and so accordingly `d[33]` returns `!`.
    Note in particular that the space character `chr(32)` becomes `d[32]`, which
    returns 'Ġ'.

    For unprintable characters, the function shifts takes the integer representing
    the Unicode code point of that character (returned by the Python `ord`) function
    and shifts it by 256. For example, `ord(" ")` returns `32`, so the the space character
    ' ' is shifted to `256 + 32`. Since `chr(256 + 32)` returns `Ġ`, we use that as the
    string representation of the space.

    This function can simplify the BPE implementation and makes it slightly easier to
    manually inspect the generated merges after they're serialized to a file.
    """
    # These 188 integers can used as-is, since they are not whitespace or control characters.
    # See https://www.ssec.wisc.edu/~tomw/java/unicode.html.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining integers 68 integers.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d

class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.merges = merges
        if special_tokens:
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)
        else:
            self.special_tokens = None
        self.vocab_inv = {v: k for k, v in self.vocab.items()}
        self.merges_dict = {pair: i for i, pair in enumerate(merges)}

        self.byte_encoder = gpt2_bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
 
    def write_files(self, vocab_filepath: str, merges_filepath: str):
        # write vocab and merges
        with open(vocab_filepath, "w", encoding="utf-8") as f:
            data = {''.join(self.byte_encoder[token] for token in v): k for k, v in self.vocab.items()}
            json.dump(data, f, ensure_ascii=False, indent=2)

        with open(merges_filepath, "w", encoding="utf-8") as f:
            for merge in self.merges:
                token1 = ''.join(self.byte_encoder[token] for token in merge[0])
                token2 = ''.join(self.byte_encoder[token] for token in merge[1])
                f.write(token1 + " " + token2 + "\n")
    
    @classmethod
    def from_files(
        cls, 
        vocab_filepath: str, 
        merges_filepath: str, 
        special_tokens: list[str] | None = None
    ):
        byte_encoder = gpt2_bytes_to_unicode()
        byte_decoder = {v: k for k, v in byte_encoder.items()}
        vocab = {}
        merges = []

        with open(vocab_filepath, encoding="utf-8") as f:
            gpt2_reference_vocab = json.load(f)
            vocab = {
                gpt2_vocab_index: bytes([byte_decoder[token] for token in gpt2_vocab_item])
                for gpt2_vocab_item, gpt2_vocab_index in gpt2_reference_vocab.items()
            }

        with open(merges_filepath, encoding="utf-8") as f:
            gpt2_reference_merges = [tuple(line.rstrip().split(" ")) for line in f]
            merges = [
                (
                    bytes([byte_decoder[token] for token in merge_token_1]),
                    bytes([byte_decoder[token] for token in merge_token_2]),
                )
                for merge_token_1, merge_token_2 in gpt2_reference_merges
            ]
        return Tokenizer(vocab, merges, special_tokens)

    def encode_single(self, text: str, token_ids: list[int]):
        parts = [text]
        if self.special_tokens:
            parts = re.split("(" + "|".join(map(re.escape, self.special_tokens)) + ")", text)

        for part in parts:
            if self.special_tokens and part in self.special_tokens:
                token_ids.append(self.vocab_inv[part.encode("utf-8")])
                continue
            for match in re.finditer(PAT, part):
                str = match.group()

                tokens = list(str_to_tuple_bytes(str))

                while True:
                    pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
                    valid_pairs = [(p, self.merges_dict[p]) for p in pairs if p in self.merges_dict]
                    if not valid_pairs:
                        break
                    best = min(valid_pairs, key=lambda x: x[1])[0]
                    i = pairs.index(best)
                    tokens = tokens[:i] + [tokens[i] + tokens[i+1]] + tokens[i+2:]

                token_ids.extend([self.vocab_inv[token] for token in tokens])
        return token_ids

    def encode(
        self, 
        text: str
    ) -> list[int]:
        token_ids: list[int] = []
        self.encode_single(text, token_ids)
        return token_ids

    def decode(
        self, 
        ids: list[int]
    ) -> str:
        res: bytes = b""
        for id in ids:
            res += self.vocab[id]
        return res.decode("utf-8", errors="replace")

    
    def encode_iterable(
        self, 
        iterable: Iterable[str]
    ) -> Iterator[int]:
        token_ids = []
        for word in iterable:
            self.encode_single(word, token_ids)
        return iter(token_ids)

class DataSetTranslate:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def call(self, dataset_path: str, output_path: str):
        token_ids = []
        with open(dataset_path, "r") as f:
            for id in self.tokenizer.encode_iterable(f):
                token_ids.append(id)
        numpy_array = np.array(token_ids, dtype=np.uint16)
        with open(output_path, "wb") as f:
            numpy_array.tofile(f)
