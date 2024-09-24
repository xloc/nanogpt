import torch
from torch import nn
import torch.nn.functional as F

from nanogpt.dataloader import DataLoader
from nanogpt.model import GPT
from nanogpt.config import *
from nanogpt.util import print_time


def main():
    model = GPT()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for x, y in DataLoader(batch_size=2, sequence_length=1024, n_cache=11):
        optimizer.zero_grad()

        for _ in range(n_batch_per_backward):
            with print_time():
                y_pred = model(x)
                loss = F.cross_entropy(y_pred.view(-1, n_vocab), y.view(-1))
                loss.backward()
            print(loss.item())

        optimizer.step()


if __name__ == "__main__":
    main()
