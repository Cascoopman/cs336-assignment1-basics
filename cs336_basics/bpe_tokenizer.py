import json
import os
from collections import defaultdict
from collections.abc import Iterable, Iterator
from typing import BinaryIO
import regex as re


def init_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    """Creates the initial vocabulary containing all possible bytes and the special tokens."""
    byte_range = 1 << 8
    vocab = {i: bytes([i]) for i in range(byte_range)}

    for i, st in enumerate(special_tokens):
        vocab[byte_range + i] = st.encode("utf-8")

    return vocab


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


def run_pretokenization(
    chunk: str, special_tokens: list[str] | None = None
) -> dict[tuple[bytes], int]:
    """Use the GPT-2 regex-based pretokenizer to split chunks into pre-tokens and return the bytes.

    Args:
        chunk: The piece of text that must be pre-tokenized
        special_tokens: A list of string special tokens to be added to the tokenizer vocabulary.
            These tokens should not be split into multiple tokens. If these special tokens occur
            in the `input_path`, they are treated as any other string.

    Returns:
        counts: The mapping between the pre-token strings and their counts
    """
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    counts = defaultdict(int)

    if special_tokens:
        # Sort special tokens by length (longest first) to handle overlapping tokens
        sorted_special_tokens = sorted(special_tokens, key=len, reverse=True)
        special_tokens_pattern = "|".join(re.escape(token) for token in sorted_special_tokens)

        for part in re.splititer(f"({special_tokens_pattern})", chunk):
            if part in special_tokens:
                counts[tuple(part.encode("utf-8"))] += 1
            else:
                for match in re.finditer(PAT, part):
                    counts[tuple(match.group(0).encode("utf-8"))] += 1
    else:
        for match in re.finditer(PAT, chunk):
            counts[tuple(match.group(0).encode("utf-8"))] += 1

    return counts


def get_pretokenization_counts(
    input_path: str | os.PathLike, special_tokens: list[str]
) -> dict[tuple[bytes], int]:
    """Chunks the input file and returns the aggregated document pre-token frequency.

    Args:
        input_path: Path to the input file.
        special_tokens: A list of special tokens to handle specially.

    Returns:
        A dictionary mapping pre-tokens (as tuples of bytes) to their frequencies.
    """
    # TODO parallelize this by sending each start/end pair to a set of processes.
    # with open(input_path, "rb") as f:
    #     NUM_PROCESSES = 4
    #     boundaries = find_chunk_boundaries(
    #         file=f, desired_num_chunks=NUM_PROCESSES, split_special_token=b"<|endoftext|>"
    #     )
    #
    #     for start, end in zip(boundaries[:-1], boundaries[1:]):
    #         f.seek(start)
    #         chunk = f.read(end - start).decode("utf-8", errors="ignore")
    #
    #         # Run pre-tokenization on your chunk and store the counts for each pre-token
    #         result = run_pretokenization(chunk, special_tokens)
    #
    #         pretoken_counts = dict(Counter(pretoken_counts) + Counter(result))

    total_counts = defaultdict(int)

    with open(input_path, encoding="utf-8") as f:
        for line in f:
            chunk_counts = run_pretokenization(line.strip(), special_tokens)
            for pretoken, count in chunk_counts.items():
                total_counts[pretoken] += count

    return total_counts


def compute_merges(
    pretoken_counts: dict[tuple[bytes], int], vocab: dict[int, bytes], vocab_size: int
) -> list[tuple[bytes, bytes]]:
    """
    Compute BPE merges from pre-tokenization counts.

    Args:
        pretoken_counts: A dictionary mapping pre-tokens to their frequencies.
        vocab: The current vocabulary.
        merges: The current list of merges.

    Returns:
        A list of merge operations.
    """
    merges = []

    while len(vocab) < vocab_size:
        # Count pairs
        pair_counts = defaultdict(int)
        for pretoken, count in pretoken_counts.items():
            if len(pretoken) < 2:
                continue
            for i in range(len(pretoken) - 1):
                pair = (pretoken[i], pretoken[i + 1])
                pair_counts[pair] += count

        if not pair_counts:
            break

        # Find the most frequent pair
        most_frequent_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))[0]
        merges.append(most_frequent_pair)

        # Add the new token to vocabulary
        new_token = most_frequent_pair[0] + most_frequent_pair[1]
        vocab[len(vocab)] = new_token

        # Update pretoken_counts to reflect the merge
        new_pretoken_counts = defaultdict(int)
        for pretoken, count in pretoken_counts.items():
            if len(pretoken) < 2:
                new_pretoken_counts[pretoken] = count
                continue

            new_pretoken = []
            i = 0
            while i < len(pretoken):
                if i < len(pretoken) - 1 and (pretoken[i], pretoken[i + 1]) == most_frequent_pair:
                    new_pretoken.append(new_token)
                    i += 2
                else:
                    new_pretoken.append(pretoken[i])
                    i += 1
            new_pretoken_counts[tuple(new_pretoken)] = count

        pretoken_counts = new_pretoken_counts

    return merges


def train_bpe_tokenizer(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str] | None = None,
    num_workers: int = 1,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer on a text file.

    Args:
        input_path: Path to the input text file.
        vocab_size: The desired vocabulary size.
        special_tokens: A list of special tokens to include in the vocabulary.
        num_workers: Number of worker processes to use.

    Returns:
        A tuple of (vocab, merges) where vocab is a dictionary mapping token IDs to bytes,
        and merges is a list of merge operations.
    """
    if special_tokens is None:
        special_tokens = []

    # Initialize vocabulary with all bytes and special tokens
    vocab = init_vocab(special_tokens)

    # Get pre-tokenization counts
    pretoken_counts = get_pretokenization_counts(input_path, special_tokens)

    # Compute merges
    merges = compute_merges(pretoken_counts, vocab, vocab_size)

    return vocab, merges


def merge_token(token: str, merges: list[tuple[bytes, bytes]]) -> list[bytes]:
    """
    Apply BPE merges to a token.

    Args:
        token: The token to merge.
        merges: The list of merge operations.

    Returns:
        A list of merged tokens.
    """
    token_bytes = token.encode("utf-8")
    token_list = [bytes([b]) for b in token_bytes]

    for merge in merges:
        if len(token_list) < 2:
            break

        new_token_list = []
        i = 0
        while i < len(token_list):
            if i < len(token_list) - 1 and (token_list[i], token_list[i + 1]) == merge:
                new_token_list.append(merge[0] + merge[1])
                i += 2
            else:
                new_token_list.append(token_list[i])
                i += 1
        token_list = new_token_list

    return token_list


class BytePairEncoder:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

    @classmethod
    def from_files(
        cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None
    ) -> "BytePairEncoder":
        with open(vocab_filepath) as f:
            vocab = json.load(f)
        with open(merges_filepath) as f:
            merges = []
            for line in f:
                line = line.strip()
                if line and len(line.split()) == 2:
                    parts = line.split()
                    merges.append((parts[0].encode("utf-8"), parts[1].encode("utf-8")))
        return cls(vocab, merges, special_tokens)

    def encode(
        self,
        text: str,
    ) -> list[int]:
        """
        Encode text into a sequence of token IDs.

        Args:
            text: The text to encode.

        Returns:
            A list of token IDs.
        """
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        pretokens: list[str] = []

        if self.special_tokens:
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            special_tokens_pattern = "|".join(re.escape(token) for token in sorted_special_tokens)

            for part in re.splititer(f"({special_tokens_pattern})", text):
                if part in self.special_tokens:
                    pretokens.append(part)
                else:
                    for match in re.finditer(PAT, part):
                        pretokens.append(match.group(0))
        else:
            for match in re.finditer(PAT, text):
                pretokens.append(match.group(0))

        # Apply BPE merges to each pre-token
        merged_tokens = []
        for token in pretokens:
            if self.special_tokens and token in self.special_tokens:
                merged_tokens.append(token.encode("utf-8"))
            else:
                merged_token = merge_token(token=token, merges=self.merges)
                merged_tokens.extend(merged_token)

        # Convert merged tokens to token IDs
        inv_vocab = {v: k for k, v in self.vocab.items()}
        encoded_tokens = []
        for token_bytes in merged_tokens:
            encoded_tokens.append(inv_vocab[token_bytes])
        return encoded_tokens

    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.

        Args:
            ids: A list of token IDs.

        Returns:
            The decoded text.
        """
        token_bytes = []
        for token_id in ids:
            token_bytes.append(self.vocab[token_id])

        # Concatenate all bytes and decode to UTF-8
        result_bytes = b"".join(token_bytes)
        return result_bytes.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Encode an iterable of strings into a sequence of token IDs.

        Args:
            iterable: An iterable of strings.

        Returns:
            An iterator of token IDs.
        """
        for text in iterable:
            yield from self.encode(text)
