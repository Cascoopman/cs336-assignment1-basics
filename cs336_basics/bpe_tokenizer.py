import os
from collections import Counter, defaultdict
from typing import BinaryIO
import regex as re


def init_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    """Creates the initial vocabulary containing all possible bytes and the special tokens."""
    byte_range = 1 << 8
    vocab = {i: i.to_bytes(length=1) for i in range(byte_range)}

    for i, st in enumerate(special_tokens):
        vocab[byte_range + i] = st.encode(encoding="utf-8")

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


def run_pretokenization(chunk: str, special_tokens: list[str]) -> dict[tuple[bytes], int]:
    """Use the GPT-2 regex-based pretokenizer to split chunks into pre-tokens and return the bytes.

    Args:
        chunk: The piece of text that must be pre-tokenized
        special_tokens: A list of string special tokens to be added to the tokenizer vocabulary.
            These tokens should not be split into multiple tokens. If these special tokens occur
            in the `input_path`, they are treated as any other string.

    Returns:
        counts: The mapping between the pre-token strings and their counts
    """
    ESC = "".join(re.escape(tok) for tok in special_tokens)

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    counts = defaultdict(int)

    # Split chunks into docs based on special tokens
    for doc in re.split(ESC, chunk):
        # Split docs into pretokens
        for match in re.finditer(PAT, doc):
            counts[tuple(match.group(0).encode("utf-8"))] += 1

    return counts


def get_pretokenization_counts(
    input_path: str | os.PathLike, special_tokens: list[str]
) -> dict[tuple[bytes], int]:
    """Chunks the input file and returns the aggregated document pre-token frequency."""
    # TODO parallelize this by sending each start/end pair to a set of processes.

    pretoken_counts = {}

    with open(input_path, "rb") as f:
        NUM_PROCESSES = 4
        boundaries = find_chunk_boundaries(
            file=f, desired_num_chunks=NUM_PROCESSES, split_special_token=b"<|endoftext|>"
        )

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")

            # Run pre-tokenization on your chunk and store the counts for each pre-token
            result = run_pretokenization(chunk, special_tokens)

            pretoken_counts = dict(Counter(pretoken_counts) + Counter(result))

    return pretoken_counts


def compute_merges(
    pretoken_counts: dict[tuple[bytes], int], vocab: dict[int, bytes], vocab_size: int
) -> list[tuple[bytes, bytes]]:
    """Iteratively counts the bytepairs and merges the most frequent and lexographical greatest pairs."""
    merges: list[tuple[bytes, bytes]] = []

    while len(vocab) < vocab_size:
        pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)

        # Count byte pairs
        for seq, cnt in pretoken_counts.items():
            for i in range(len(seq) - 1):
                pair_counts[(seq[i], seq[i + 1])] += cnt

        if not pair_counts:
            break

        # Select most frequent pair, cutting ties by lexographical greatest
        merge_pair = max(pair_counts, key=lambda k: (pair_counts[k], k))
        a, b = merge_pair

        # Python represents bytes objects with only one element as an integer
        a_bytes = a.to_bytes(1) if isinstance(a, int) else a
        b_bytes = b.to_bytes(1) if isinstance(b, int) else b

        # Append to the vocab
        new_token = a_bytes + b_bytes
        vocab[len(vocab)] = new_token
        merges.append((a_bytes, b_bytes))

        # Rebuild pretoken counts
        new_pretoken_counts: dict[tuple[bytes, ...], int] = defaultdict(int)

        for seq, cnt in pretoken_counts.items():
            i = 0
            out: list[bytes] = []
            while i < len(seq):
                if i + 1 < len(seq) and seq[i] == a and seq[i + 1] == b:
                    out.append(new_token)
                    i += 2
                else:
                    out.append(seq[i].to_bytes(1) if isinstance(seq[i], int) else seq[i])
                    i += 1
            new_pretoken_counts[tuple(out)] += cnt

        pretoken_counts = new_pretoken_counts

    return merges


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
    vocab = init_vocab(special_tokens=special_tokens)

    pretoken_counts = get_pretokenization_counts(
        input_path=input_path, special_tokens=special_tokens
    )

    merges = compute_merges(pretoken_counts=pretoken_counts, vocab=vocab, vocab_size=vocab_size)

    return vocab, merges


if __name__ == "__main__":
    vocab, merges = train_bpe_tokenizer(
        input_path="data/TinyStoriesV2-GPT4-valid.txt",
        vocab_size=300,
        special_tokens=["<|endoftext|>"],
    )
    print("Vocab: ", vocab, "\n" * 2, "Merges: ", merges)
