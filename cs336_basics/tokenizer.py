import os
from typing import BinaryIO
import regex as re
from collections import defaultdict
import time
import json
from typing import Iterator, Iterable
import multiprocessing as mp

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
    num_processes = 4
    tasks : list[tuple[int, int]]


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
    for match in count:
        for idx in range(len(match) - 1):
            test[(match[idx], match[idx + 1])] += count[match]
    while len(vocab) < vocab_size:
        # merge
        best_key = max(test.items(), key=lambda kv: (kv[1], kv[0]))[0]

        for match in list(count.keys()):
            change = False
            result = []
            for idx in range(len(match) - 1):
                # 只更新命中的
                if match[idx] == best_key[0] and match[idx + 1] == best_key[1]:
                    change = True
                    break
            if change:
                idx = 0
                while idx < len(match):
                    if idx < (len(match) - 1) and match[idx] == best_key[0] and match[idx + 1] == best_key[1]:
                        result.append(match[idx] + match[idx + 1])
                        idx += 2
                    else:
                        result.append(match[idx])
                        idx += 1
                # 更新新的match和result
                for idx in range(len(match) - 1):
                    test[(match[idx], match[idx + 1])] -= count[match]
                for idx in range(len(result) - 1):
                    test[(result[idx], result[idx + 1])] += count[match]
                count[tuple(result)] += count.pop(match)
        merges.append(best_key)
        vocab[len(vocab)] = (best_key[0] + best_key[1])
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

    def encode_single(self, text: str, token_ids: list[int], cache: dict[str, list[int]]):
        parts = [text]
        if self.special_tokens:
            parts = re.split("(" + "|".join(map(re.escape, self.special_tokens)) + ")", text)

        for part in parts:
            if self.special_tokens and part in self.special_tokens:
                token_ids.append(self.vocab_inv[part.encode("utf-8")])
                continue
            for match in re.finditer(PAT, part):
                str = match.group()

                if str not in cache:
                    matchBytes = str_to_tuple_bytes(str)

                    if matchBytes in self.vocab_inv:
                        cache[str] = self.vocab_inv[matchBytes]
                    else:
                        for merge in self.merges:
                            idx = 0
                            result = []
                            change = False
                            while idx < len(matchBytes):
                                if idx < len(matchBytes) - 1 and matchBytes[idx] == merge[0] and matchBytes[idx + 1] == merge[1]:
                                    change = True
                                idx += 1
                            idx = 0
                            if change:
                                while idx < len(matchBytes):
                                    if idx < len(matchBytes) - 1 and matchBytes[idx] == merge[0] and matchBytes[idx + 1] == merge[1]:
                                        result.append(matchBytes[idx] + matchBytes[idx + 1])
                                        idx += 2
                                    else:
                                        result.append(matchBytes[idx])
                                        idx += 1
                                matchBytes = result

                        token_id: list[int] = []
                        for b in matchBytes:
                            token_id.append(self.vocab_inv[b])
                        
                        cache[str] = token_id

                token_ids.extend(cache[str])

    def encode(
        self, 
        text: str
    ) -> list[int]:
        token_ids: list[int] = []
        cache: dict[str, list[int]] = {}
        self.encode_single(text, token_ids, cache)
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
        cache: dict[str, list[int]] = {}
        for word in iterable:
            self.encode_single(word, token_ids, cache)
        return iter(token_ids)