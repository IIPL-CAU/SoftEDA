# Standard Library Modules
import pickle
# 3rd-party Modules
from tqdm.auto import tqdm
# Pytorch Modules
from torch.utils.data.dataset import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_path:str) -> None:
        super(CustomDataset, self).__init__()
        with open(data_path, 'rb') as f:
            data_ = pickle.load(f)

        self.data_list = []
        self.vocab_size = data_['vocab_size']
        self.num_classes = data_['num_classes']
        self.pad_token_id = data_['pad_token_id']

        for idx in tqdm(range(len(data_['input_ids'])), desc=f'Loading data from {data_path}'):
            self.data_list.append({
                'input_ids': data_['input_ids'][idx],
                'attention_mask': data_['attention_mask'][idx],
                'token_type_ids': data_['token_type_ids'][idx],
                'labels': data_['labels'][idx],
                'soft_labels': data_['soft_labels'][idx],
                'index': idx
            })

        del data_

    def __getitem__(self, idx:int) -> dict:
        return self.data_list[idx]

    def __len__(self) -> int:
        return len(self.data_list)
