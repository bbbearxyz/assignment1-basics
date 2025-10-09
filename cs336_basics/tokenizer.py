import os
from typing import BinaryIO
import regex as re
from collections import defaultdict
import time
import json
from typing import Iterator, Iterable
import multiprocessing as mp
import heapq

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

    heap = []
    for key, value in test.items():
        heapq.heappush(heap, (-value, key))

    mid_time = time.time()
    time_find_max = 0
    while len(vocab) < vocab_size:
        # merge
        first = time.time()

        # 懒删除
        while heap:
            freq, pair = heapq.heappop(heap)
            freq = -freq
            if test[pair] == freq:
                break

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
                    test[new_token] -= count[match]
                    heapq.heappush(heap, (-test[new_token], new_token))
                for idx in range(len(result) - 1):
                    test[new_token] += count[match]
                    heapq.heappush(heap, (-test[new_token], new_token))
                count[tuple(result)] += count.pop(match)
        merges.append(pair)
        vocab[len(vocab)] = new_token
    
    end_time = time.time()
    print("pre token cost", mid_time - start_time)
    print("merge cost", end_time - mid_time)
    print("find max", time_find_max)
    return (vocab, merges)

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


    def from_files(
        cls, 
        vocab_filepath: str, 
        merges_filepath: str, 
        special_tokens: list[str] | None = None
    ):
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            print(data)
            raise ValueError("xxx")

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