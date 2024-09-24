import torch
from nanogpt.token_chunks import generate_token_chunks
from nanogpt.config import device
from nanogpt.util import cache_iter


class DataLoader:
    def __init__(self, batch_size: int, sequence_length: int, n_cache: int = 3):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        cached = cache_iter(n_cache)(generate_token_chunks)
        self.iter = cached(batch_size * sequence_length + 1, 1)

    def __iter__(self):
        return self

    def __next__(self):
        batch = next(self.iter)
        batch = torch.from_numpy(batch).long().to(device)
        x = batch[:-1].view(self.batch_size,  -1)
        y = batch[1:].view(self.batch_size, -1)
        return x, y

    def __len__(self):
        n_token = 10_000_000_000  # fineweb-edu/CC-MAIN-2024-10 has 10B tokens
        return n_token // (self.batch_size * self.sequence_length)


if __name__ == "__main__":
    loader = DataLoader(32, 1024)
    for xy, _ in zip(loader, range(2)):
        x, y = xy
        print(x.shape, y.shape)
