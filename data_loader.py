import torch
from torch.utils.data import Dataset
class TalkDataset(Dataset):
    def __init__(self, mode, list_of_dict):
        assert mode in ["train", "test"]
        self.mode = mode
        self.list_of_dict = list_of_dict
            
    def __getitem__(self,idx):
        inputid = self.list_of_dict[idx]['input_ids']
        tokentype = self.list_of_dict[idx]['seg']
        attentionmask = self.list_of_dict[idx]['att']
        id = self.list_of_dict[idx]['article_id']
        inputid = torch.tensor(inputid)
        tokentype = torch.tensor(tokentype)
        attentionmask = torch.tensor(attentionmask)
        if (self.mode == "test"):
            return inputid, tokentype, attentionmask, id
        elif (self.mode == "train"):
            type_label = self.list_of_dict[idx]['type_label']
            BIO_label = self.list_of_dict[idx]['BIO_label']
            type_label = torch.tensor(type_label)
            BIO_label = torch.tensor(BIO_label)
            return inputid, tokentype, attentionmask, type_label, BIO_label, id
    
    def __len__(self):
        return len(self.list_of_dict)