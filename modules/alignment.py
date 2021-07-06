import torch
import sys
sys.path.append("..")
import param
import torch.nn as nn
from transformers import BartTokenizer, BartModel
from torch.autograd import Function
import torch.nn.functional as F

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

class Discriminator(nn.Module):
    """ This is A for M3, M4, M5"""
    def __init__(self):
        """Init discriminator."""
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(param.hidden_size, param.intermediate_size),
            nn.LeakyReLU(),
            nn.Linear(param.intermediate_size, param.intermediate_size),
            nn.LeakyReLU(),
            nn.Linear(param.intermediate_size, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        """Forward the discriminator."""
        out = self.layer(x)
        return out

class DomainClassifier(nn.Module):
    def __init__(self, dropout=0.1):
        super(DomainClassifier, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(param.hidden_size, 1)
        self.sig = nn.Sigmoid()
        self.apply(self.init_bert_weights)

    def forward(self, x, alpha):
        x = self.dropout(x)
        x = ReverseLayerF.apply(x, alpha)
        x = self.classifier(x)
        out = self.sig(x)
        return out

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


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

class BartDecoder(nn.Module):
    def __init__(self):
        super(BartDecoder,self).__init__()
        self.decoder=BartModel.from_pretrained('facebook/bart-base').decoder
        self.config=self.decoder.config
        
        self.lm_head1 = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        
    def forward(self,
        input_ids=None,
        decoder_inputs_embeds=None,
        decoder_input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None):
        
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(input_ids, self.config.pad_token_id, self.config.decoder_start_token_id)
                
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            
            output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
            
            use_cache = use_cache if use_cache is not None else self.config.use_cache
            
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
            
            
            output = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
            
            logits1=self.lm_head1(output[0])
            return output,logits1
 

