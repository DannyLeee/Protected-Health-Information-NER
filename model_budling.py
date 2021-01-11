""" model budling """
from transformers import AutoModel
import torch
import torch.nn as nn

class PHI_NER(nn.Module):
    def __init__(self, PRETRAINED_LM):
        super(PHI_NER, self).__init__()
        self.bert = AutoModel.from_pretrained(PRETRAINED_LM, output_hidden_states=True)
        # self.bert = XLNetModel.from_pretrained(PRETRAINED_LM, output_hidden_states=True, mem_len=1024)
        self.bert.resize_token_embeddings(self.bert.config.vocab_size + 9)
        self.type_classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 19),
            # nn.ReLU(),
            # nn.Dropout(0.1),
            # nn.Linear(128, 19)
        )
        
        self.BIO_classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 3),
            # nn.ReLU(),
            # nn.Dropout(0.1),
            # nn.Linear(128, 3)
        )
        self.softmax = nn.Softmax(-1)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids)

        # pooler_output = torch.rand(outputs[0].shape[0], 1, outputs[0].shape[2]).to(outputs[0].device)
        # for i in range(outputs[0].shape[1]):
        #     o = self.bert.pooler(outputs[0][:,i,:].unsqueeze(1))
        #     pooler_output = torch.cat((pooler_output ,o.unsqueeze(1)), 1)
        # pooler_output = pooler_output[:,1:,:]
        
        type_ = self.type_classifier(outputs[0]) # 512*HIDDEN_SIZE word vectors
        type_ = self.softmax(type_)
        BIO = self.BIO_classifier(outputs[0]) # 512*HIDDEN_SIZE word vectors
        BIO = self.softmax(BIO)
                
        outputs = (type_, BIO, ) + outputs[2:]
        return outputs