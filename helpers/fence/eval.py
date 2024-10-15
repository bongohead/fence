import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from torch.utils.data import DataLoader

import math
import numpy as np

from IPython.core.display import HTML, Markdown, display
from termcolor import colored

from helpers.misc import is_notebook
from helpers.phi3.phi3 import _prepare_4d_causal_attention_mask, apply_rotary_pos_emb

import pandas as pd 

@torch.no_grad()
def generate_fence(model, tokenizer, prompt, echo_output = True, max_tokens = 128, device = 'cuda'):
    """
    Runs a forward pass and stores FENCE-relevant intermediate hidden states

    Returns a dictionary with keys:
        - `text`: The decoded output text, as a list
        - `hk1s`: The first residual stream output
        - `hk2s`: The final residual stream output
        - `hksas`: The hidden state outputs of the SA component
        - `hkmlps`: The hidden state outputs of the MLP component
    """
    model.eval()
    generated_tokens = 0
    
    input_ids_0 = tokenizer(prompt, return_tensors = 'pt').to(device)['input_ids']
    input_ids = input_ids_0

    while True:
        embeds_output = model.model.embed_tokens(input_ids)
        hidden_state = embeds_output
        
        B, N, D = embeds_output.shape
        H = 32
        Dh = int(D/H)
        
        position_ids = torch.arange(0, N, dtype=torch.long, device=device).unsqueeze(0).view(-1, N) # Create position IDs
        
        # Flash attention = use default attention mask 2d
        if model.model._attn_implementation == 'flash_attention_2':
            attention_mask = None
        # Non flash-attention: Make a triangular attention mask to hide right context
        else:
            attention_mask = _prepare_4d_causal_attention_mask(None, (B, N), embeds_output, 0, sliding_window = model.model.config.sliding_window) 

        saved_sa_outputs = []
        saved_hkrs = []
        saved_mlp_outputs = []
        saved_hks = []
        ### Transformer Blocks ###
        for i, layer in enumerate(model.model.layers):            

            residual = hidden_state
            sa_input = layer.input_layernorm(hidden_state)
            
            ### SA ###
            sa_module = layer.self_attn
            # sa_output = sa_module(sa_input, attention_mask, position_ids)[0]
            qkv = sa_module.qkv_proj(sa_input)
            queries = qkv[:, :, :D].view(B, N, H, Dh).transpose(1, 2)
            keys = qkv[:, :, D:2*D].view(B, N, H, Dh).transpose(1, 2)
            values = qkv[:, :, 2*D:].view(B, N, H, Dh).transpose(1, 2)

            if model.model._attn_implementation == 'flash_attention_2':     
                # Flash attention requires the input to have the shape B x N x Dh x D           
                # Because the input can be padded, the absolute sequence length depends on the max position id.
                rotary_seq_len = max(N, position_ids[:, -1].max().item()) + 1
                cos, sin = sa_module.rotary_emb(values, position_ids, seq_len = rotary_seq_len)
                queries, keys = apply_rotary_pos_emb(queries, keys, cos, sin, position_ids)
                ################## # Reshape to the expected shape for Flash Attention
                queries = queries.transpose(1, 2)
                keys = keys.transpose(1, 2)
                values = values.transpose(1, 2)
                ###################
                sa_output = sa_module._flash_attention_forward(queries, keys, values, attention_mask, N)
                sa_output = sa_output.reshape(B, N, D).contiguous()
            else:    
                cos, sin = sa_module.rotary_emb(values, position_ids, seq_len = N)
                queries, keys = apply_rotary_pos_emb(queries, keys, cos, sin, position_ids)
                attn_weights = torch.matmul(queries, keys.transpose(2, 3))/math.sqrt(Dh)  # Should be shape B x H x N x N
                attn_weights = attn_weights + attention_mask # Attemtion mask is upper triangular of negative infinity
                attn_weights = F.softmax(attn_weights, dim = -1, dtype = torch.float32).to(values.dtype)
                sa_output = torch.matmul(attn_weights, values) # B x H x N x D/H
                sa_output = sa_output.transpose(1, 2).contiguous() # Reorder into B x N x H x D/H
                sa_output = sa_output.reshape(B, N, D) # Concatenate vertically back into B x N x D
    
            # Finally post-concatenation linear layer
            sa_output = sa_module.o_proj(sa_output)

            saved_sa_outputs.append(sa_output[0, :, :].detach())
            
            ### add residual -> store residual -> layernorm -> mlp -> add residual
            hidden_state = residual + sa_output
            residual = hidden_state
            saved_hkrs.append(hidden_state[0, :, :].detach())

            hidden_state = layer.post_attention_layernorm(hidden_state)
            ## MLP            
            up_state = layer.mlp.gate_up_proj(hidden_state) # B x N x (2I, I = intermediate MLP dimension)
            gate, up_state = up_state.chunk(2, dim = -1) # B x N x I
            up_state = up_state * layer.mlp.activation_fn(gate)  # Elementwise
            hidden_state = layer.mlp.down_proj(up_state) # Back to B x N x D
            ## End MLP
            
            saved_mlp_outputs.append(hidden_state[0, :, :].detach())

            hidden_state = residual + hidden_state

            saved_hks.append(hidden_state[0, :, :].detach())
                
        hidden_state = model.model.norm(hidden_state)
        logits = model.lm_head(hidden_state)

        # Get argmax tokens + concatenate onto previous tokens
        output_token = torch.argmax(F.softmax(logits.squeeze(), dim = 1), dim = 1)[-1]
        input_ids = torch.cat((input_ids, output_token.view(1, 1)), dim = 1)

        # Break while loop if EOS or generation > max tokens
        generated_tokens = generated_tokens + 1
        if output_token in [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|end|>")] or generated_tokens >= max_tokens:
            break

    # Use it on the last pass only
    all_hksas = [h.cpu().to(torch.float16).numpy() for h in saved_sa_outputs]
    all_hkrs = [h.cpu().to(torch.float16).numpy() for h in saved_hkrs]
    all_hkmlps = [h.cpu().to(torch.float16).numpy() for h in saved_mlp_outputs]
    all_hks = [h.cpu().to(torch.float16).numpy() for h in saved_hks]

    final_output = input_ids.squeeze()
    decoded_text = tokenizer.batch_decode(final_output)
    decoded_output = tokenizer.decode(final_output[input_ids_0.size()[1]:])

    if echo_output:
        if is_notebook():
            display(HTML(
                '<div style="padding: 1rem 2rem; background-color:honeydew">' + 
                    '<h4>Modified model output</h4>' + 
                    '<span style="color:green">' + tokenizer.batch_decode(input_ids_0)[0][3:] + '</span> ' + 
                    '<span style="color:red">' + decoded_output + '</span>' +
                '</div>'
            ))
        else:
            print(colored(tokenizer.batch_decode(input_ids_0)[0][3:], 'green'), colored(tokenizer.decode(final_output[input_ids_0.size()[1]:]), 'red'))
            
    return {
        'text': decoded_text,
        'hkrs': all_hkrs, 
        'hks': all_hks,
        'hksas': all_hksas,
        'hkmlps': all_hkmlps
    }



@torch.no_grad()
def get_logit_lens(model, tokenizer, hidden_state, top_k):
    """
    Feed an intermediate B x N x D hidden state block into the RMSNorm and LM Head to get the logit lens output.

    Params:
        @model: The Phi-3 model object. Must have model.model.norm as the final post-transformer norm layer, and model.lm_head as the LM head.
        @tokenizer: A tokenizer
        @hidden_state: A B x N x D hidden state object to feed through the logit lens.
        @top_k: The top k probability tokens to return.
    Returns:
        A dataframe with columns input_index, token_rank, token, and probability.
    """
    
    #### Finish Forward Pass #####
    # RMS Norm
    hidden_state = model.model.norm(hidden_state)

    # Run LM head
    logits = model.lm_head(hidden_state).float() # B x N x D
    #### End Forward Pass #####

    last_token_logits = logits[:, -1, :]
    
    probabilities = F.softmax(last_token_logits, dim = -1)

    # Optionally, map the probabilities to tokens using the tokenizer's vocabulary
    top_probabilities, top_indices = torch.topk(probabilities, top_k, dim=-1)  # B x top_k

    res = {'input_index': [], 'token_rank': [], 'token': [], 'probability': []}

    # Iterate over batch size and fill the data
    batch_size = top_probabilities.size(0)
    for i in range(batch_size):
        for rank in range(top_k):
            # Extract token id and probability
            token_idx = top_indices[i, rank].item()
            prob = round(top_probabilities[i, rank].item(), 6)

            # Decode token using the tokenizer
            token = tokenizer.decode([token_idx])

            # Append data to the lists
            res['probability'].append(prob)
            res['input_index'].append(i)
            res['token_rank'].append(rank + 1)  # rank starts from 1
            res['token'].append(token)

    return pd.DataFrame(res)
