import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

n_embedding = 768
n_vocab = 50257
max_sequence_len = 1024
n_block = 12
n_head = 12

batch_size = 32
sequence_length = 1024
n_batch_per_backward = 1
