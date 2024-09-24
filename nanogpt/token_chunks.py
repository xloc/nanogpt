import numpy as np
from datasets import load_dataset
import tiktoken
import multiprocessing

import warnings
warnings.filterwarnings("ignore", category=UserWarning,
                        message=r"^\nThe secret `HF_TOKEN` does not exist.*")


tokenizer = tiktoken.get_encoding("gpt2")
eot = tokenizer._special_tokens['<|endoftext|>']


def worker(row):
    return tokenizer.encode(row['text'])


def generate_token_chunks(n_token: int, n_processes: int = None):
    dataset = load_dataset("HuggingFaceFW/fineweb-edu",
                           name="CC-MAIN-2024-10", split="train", streaming=True)
    pool = multiprocessing.Pool(n_processes)

    tokens = []
    for row_tokens in pool.imap(worker, dataset):
        tokens.append(eot)
        tokens.extend(row_tokens)
        if len(tokens) > n_token:
            chunk, reminder = tokens[:n_token], tokens[n_token:]
            tokens = reminder
            yield np.array(chunk, dtype=np.uint16)
    yield np.array(tokens, dtype=np.uint16)  # the last chunk


if __name__ == "__main__":
    from nanogpt.util import print_time

    with print_time():
        chunk = next(iter(generate_token_chunks(1_000_000, 1)))
        print(chunk.shape)
    with print_time():
        chunk = next(iter(generate_token_chunks(1_000_000, 4)))
        print(chunk.shape)
