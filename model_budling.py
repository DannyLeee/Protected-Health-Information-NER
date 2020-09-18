""" model budling """
from transformers import BertModel
import torch
import torch.nn as nn

PRETRAINED_LM = "hfl/chinese-bert-wwm"

class PHI_NER(nn.Module):
    def __init__(self):
        super(PHI_NER, self).__init__()
        self.bert = BertModel.from_pretrained(PRETRAINED_LM, output_hidden_states=True)
        self.type_classifier = nn.Linear(self.bert.config.hidden_size, 19) # type
        self.BIO_classifier = nn.Linear(self.bert.config.hidden_size, 3) # BIO tagging
        self.softmax = nn.Softmax(-1)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids)
        
        type_ = self.type_classifier(outputs[0]) # 512*HIDDEN_SIZE word vectors
        type_ = self.softmax(type_)
        BIO = self.BIO_classifier(outputs[0]) # 512*HIDDEN_SIZE word vectors
        BIO = self.softmax(BIO)
                
        outputs = (type_, BIO, ) + outputs[2:]
        return outputs