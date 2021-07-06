import torch
import sys
sys.path.append("..")
import param
import torch.nn as nn
from transformers import BertModel
from torch.autograd import Function
from transformers import BartTokenizer, BartModel

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        output = output
        return output, None


class BertEncoder(nn.Module):
    def __init__(self):
        super(BertEncoder, self).__init__()
        self.encoder = BertModel.from_pretrained('bert-base-multilingual-cased')
    def forward(self, x, mask=None,segment=None):
        outputs = self.encoder(x, attention_mask=mask,token_type_ids=segment)
        feat = outputs[1]
        return feat

class MLP(nn.Module):

    """ use MLP to choose feature of target data"""
    def __init__(self):
        """Init discriminator."""
        super(MLP, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(param.hidden_size, param.h_dim),
            nn.LeakyReLU(),
            nn.Linear(param.h_dim, param.hidden_size),
            nn.LeakyReLU()
        )
        
    def forward(self, x):
        """Forward the MLP."""
        out = self.layer(x)
        return out


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class BartPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
class BartEncoder(nn.Module):
    def __init__(self):
        super(BartEncoder,self).__init__()
        self.encoder=BartModel.from_pretrained('facebook/bart-base').encoder
        self.config=self.encoder.config
        
        self.lm_head1 = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.pooler = BartPooler(self.config)
    def forward(self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,):
        
        output=self.encoder(input_ids=input_ids,attention_mask=attention_mask,
                            head_mask=head_mask,inputs_embeds=inputs_embeds,
                            output_attentions=output_attentions,output_hidden_states=output_hidden_states,
                            return_dict=return_dict)

        sequence_output = output[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        
        
        logits1=self.lm_head1(output[0])
        return output,logits1,pooled_output
