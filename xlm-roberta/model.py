from transformers import AutoModel
import torch

class XLM_RoBERTa_classifier_one(torch.nn.Module):
    def __init__(self, model_checkpoint:str = 'xlm-roberta-base', dropout:float = 0.0):
        super(XLM_RoBERTa_classifier_one, self).__init__()
        self.roberta = AutoModel.from_pretrained(model_checkpoint)
        for param in self.roberta.parameters():
          param.requires_grad = False
        self.dropout = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(768, 3)
        self.softmax = torch.nn.Softmax(dim=1)
    
    def forward(self, input_ids, attention_mask):
        pooled_output = self.roberta(input_ids, attention_mask)[1]
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.softmax(linear_output)
        return final_layer