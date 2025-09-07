import os
from collections import Counter
from typing import BinaryIO
import regex as re

def train_bpe_tokenizer(
    input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # Initialize vocab with all possible bytes
    vocab = {i: i.to_bytes(1) for i in range(256)}
    
    # Add special tokens to the vocab
    for i, st in enumerate(special_tokens):
        vocab[st] = i+255

    merges = []

    # Get the pre-tokenization counts
    pretoken_counts = {}

    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")

            # Run pre-tokenization on your chunk and store the counts for each pre-token
            result = run_pretokenization(chunk, special_tokens)

            pretoken_counts = dict(Counter(pretoken_counts) + Counter(result))
    
    for k, v in pretoken_counts.items():
        if v > 20000:
            print(repr(k), v)
    
    # TODO
    quit()

    while len(vocab) <= vocab_size:

        # Compute BPE merges
        pair_counts = {}
        
        # Count all byte pairs
        for pretoken, count in pretoken_counts:
            for i in range(len(pretoken)-1):
                pair_counts[set(pretoken[i], pretoken[i+1])] += count
        
        # Merge the most frequent pair, preferring lexographical greatest
        merge_pair = max(pair_counts)
        merges.add(merge_pair)

        for token, count in pretoken_counts:
            for i in range(len(token)-1):
                if set(pretoken[i], pretoken[i+1]) == merge_pair:
                    pretoken.pop(i)
                    pretoken[i+1] = merge_pair

        # Add it to the vocab
        vocab[merge_pair] = len(vocab)

        
    return vocab, merges


def run_pretokenization(chunk: str, special_tokens: list[str]) -> dict[bytes, int]:
    """Use the GPT-2 regex-based pretokenizer to split chunks into pre-tokens and return the bytes.
    
    Args:
        chunk (str): The piece of text that must be pre-tokenized
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        counts (dict[bytes, int]): The mapping between the pre-tokens in bytes and their counts
    """
    ESC = "".join(re.escape(tok) for tok in special_tokens)

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""    

    counts = {}

    # Split chunks into docs based on special tokens
    for doc in re.split(ESC, chunk):
        # Split docs into pretokens
        for match in re.finditer(PAT, doc):
            try:
                counts[match.group(0)] += 1      
            except KeyError:
                counts[match.group(0)] = 1 

    return counts


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


if __name__ == "__main__":
    vocab, merges = train_bpe_tokenizer(
        input_path="data/TinyStoriesV2-GPT4-valid.txt",
        vocab_size=300,
        special_tokens=["<|endoftext|>"],
    )
    print("Vocab: ", vocab, "Merges: ", merges)