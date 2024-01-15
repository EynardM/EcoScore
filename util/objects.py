from util.imports import *

# Trainer dataset object
class TrainerDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: Dataset, tokenizer):
        self.reviews = dataset['review']
        self.rates = dataset['rate']
        self.input_ids = tokenizer(dataset['review'], padding='max_length', truncation=True, max_length=400, return_tensors='pt')['input_ids']
        self.labels = [rate + 1 for rate in self.rates]  

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'labels': self.labels[idx]
        }

    def __len__(self):
        return len(self.reviews)
