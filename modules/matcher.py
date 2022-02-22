import torch
import sys
sys.path.append("..")
import param
import torch.nn as nn
class BertClassifier(nn.Module):
    """This is the matcher when Feature Extractor is LMs (Bert etc.)"""
    def __init__(self, dropout=0.1):
        super(BertClassifier, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(param.hidden_size, param.num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, x):
        x = self.dropout(x)
        out = self.classifier(x)
        return out

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
