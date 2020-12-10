import torch
from tqdm import tqdm
PATH = "../data/tensor-data"
dataset = torch.load(f"{PATH}/EEG4DTIME_3SPLIT.pt")
for i, split in tqdm(enumerate(dataset)):
    torch.save(dataset[split], f"{PATH}/EEG4DTIME-{i}.pt")
