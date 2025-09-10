import os
from collections import Counter, defaultdict
from typing import BinaryIO
import regex as re
from bidict import bidict


def replace_all_pairs_in_keys(d, merge_pair, new_id):
    a, b = merge_pair
    merged = defaultdict(int)

    for key, count in d.items():
        i = 0
        new = []
        while i < len(key):
            if i + 1 < len(key) and key[i] == a and key[i + 1] == b:
                new.append(new_id)
                i += 2
            else:
                new.append(key[i])
                i += 1
        merged[tuple(new)] += count

    return dict(merged)


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
    # --- Initialize vocab ---
    # Add all possible bytes
    byte_range = 1 << 8
    vocab = {i.to_bytes(1): i for i in range(byte_range)}
    
    # Add special tokens to the vocab
    for i, st in enumerate(special_tokens):
        vocab[st.encode()] = i + byte_range

    # --- Get the pre-tokenization counts ---
    pretoken_counts = {}

    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # TODO parallelize this by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")

            # Run pre-tokenization on your chunk and store the counts for each pre-token
            result = run_pretokenization(chunk, special_tokens)

            pretoken_counts = dict(Counter(pretoken_counts) + Counter(result))

    # --- Compute BPE merges ---
    merges = []

    translated_counts = {}
    # Translate to ID's
    for k, v in pretoken_counts.items():
        group = tuple(b for b in k.encode())
        translated_counts[group] = v

    while len(vocab) < vocab_size:
        pair_counts = defaultdict(int)

        # Count all byte pairs
        for pretoken_ids, count in translated_counts.items():
            for i in range(len(pretoken_ids)-1):
                pair_counts[(pretoken_ids[i], pretoken_ids[i+1])] += count

        # Merge the most frequent pair, preferring lexographical greatest
        merge_pair = max(pair_counts, key=lambda k: (pair_counts[k], k))
        merges.append(merge_pair)

        # Add it to the vocab
        v = bidict(vocab)  # bytes -> id; v.inv is id -> bytes
        def add_merge(v: bidict, a_id: int, b_id: int) -> int:
            merged = v.inv[a_id] + v.inv[b_id]
            if merged in v:
                return v[merged]
            new_id = (max(v.inv) + 1) if v else 0
            vocab[merged] = new_id
            return new_id

        new_id = add_merge(v, merge_pair[0], merge_pair[1])

        # Replace the merge pair tuples with single ID
        translated_counts = replace_all_pairs_in_keys(translated_counts, merge_pair, new_id)
        
    return vocab, merges


def run_pretokenization(chunk: str, special_tokens: list[str]) -> dict[str, int]:
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

    counts = defaultdict(int)

    # Split chunks into docs based on special tokens
    for doc in re.split(ESC, chunk):
        # Split docs into pretokens
        for match in re.finditer(PAT, doc):
            counts[match.group(0)] += 1      

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
    print("Vocab: ", vocab, "\n"*2, "Merges: ", merges)