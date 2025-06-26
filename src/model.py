# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union

from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.activations import ACT2FN
from config.config import GPTConfig # Import the modified GPTConfig

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout) # config.attention_probs_dropout_prob equivalent
        self.resid_dropout = nn.Dropout(config.dropout) # config.hidden_dropout_prob equivalent

        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_attention_heads

        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if not self.flash:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("causal_mask", torch.tril(torch.ones(config.max_position_embeddings, config.max_position_embeddings))
                                        .view(1, 1, config.max_position_embeddings, config.max_position_embeddings))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None, # HF expects this
        layer_past: Optional[Tuple[torch.Tensor]] = None, # For kv caching
        head_mask: Optional[torch.Tensor] = None, # HF expects this
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ):
        B, T, C = hidden_states.size() # batch size, sequence length, embedding dimensionality (hidden_size)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(hidden_states).split(self.hidden_size, dim=2)

        k = k.view(B, T, self.num_attention_heads, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.num_attention_heads, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.num_attention_heads, self.head_dim).transpose(1, 2) # (B, nh, T, hs)

        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        present = (k, v) if use_cache else None

        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            # HF's SDPA uses a different signature if attention_mask is provided
            # For causal, we can pass is_causal=True if attention_mask is None
            # If attention_mask is not None, it should be appropriately shaped for SDPA
            is_causal_sdpa = True if attention_mask is None and T > 1 else False # T > 1 condition from HF GPTNeoX
            attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, dropout_p=self.attn_dropout.p if self.training else 0.0, is_causal=is_causal_sdpa)
            attn_weights = None # Flash attention does not return weights by default
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # Apply the causal mask
            att = att.masked_fill(self.causal_mask[:,:,:T,:T] == 0, float('-inf'))
            # Apply the attention mask (if provided by Hugging Face)
            if attention_mask is not None:
                att = att + attention_mask # HF attention masks are additive

            att = F.softmax(att, dim=-1)
            attn_weights = att # Store for output_attentions
            att = self.attn_dropout(att)
            attn_output = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        attn_output = self.resid_dropout(self.c_proj(attn_output))

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs # attn_output, present_key_value, (attn_weights)

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc    = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.bias)
        self.act = ACT2FN[config.activation_function] if isinstance(config.activation_function, str) else config.activation_function
        self.c_proj  = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout) # config.hidden_dropout_prob

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.hidden_size, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.hidden_size, bias=config.bias)
        self.mlp = MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            layer_past=layer_past,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:] # present, (attentions)

        # residual connection
        hidden_states = residual + attn_output # First residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states # Second residual

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present_key_value, (attentions)


class GPT(PreTrainedModel):
    config_class = GPTConfig # Associate with the custom config

    def __init__(self, config: GPTConfig):
        super().__init__(config)
        self.config = config # Save config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.hidden_size),
            wpe = nn.Embedding(config.max_position_embeddings, config.hidden_size),
            drop = nn.Dropout(config.dropout), # config.embd_pdrop equivalent
            h = nn.ModuleList([Block(config) for _ in range(config.num_hidden_layers)]),
            ln_f = LayerNorm(config.hidden_size, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        # self.post_init() # This is called by PreTrainedModel's __init__

        # Tie weights if specified in config (though not explicitly in this GPTConfig yet)
        # self.tie_weights() # Call this if lm_head and wte should share weights

        # Report number of parameters (optional, for info)
        # print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_input_embeddings(self):
        return self.transformer.wte

    def set_input_embeddings(self, new_embeddings):
        self.transformer.wte = new_embeddings

    def _init_weights_custom(self, module): # Renamed to avoid conflict if super()._init_weights exists
        """ Initializes weights for linear and embedding layers. """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # Special initialization for projection layers in attention, as per original code
        if isinstance(module, nn.Linear) and module.weight.shape[-1] == self.config.hidden_size and 'c_proj' in str(module): # Heuristic
             torch.nn.init.normal_(module.weight, mean=0.0, std=0.02/math.sqrt(2 * self.config.num_hidden_layers))


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None, # Not used by GPT-2 like models
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None, # Mask for attention heads
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
            device = input_ids.device
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
            device = inputs_embeds.device
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.transformer.h))

        if position_ids is None:
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length) # (1, T)

        # Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                 raise ValueError("batch_size has to be defined and > 0")
            # HF GPT2 attention mask processing
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :] # (B, 1, 1, T_to)
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers) # Prepares head_mask for per-layer application

        if inputs_embeds is None:
            inputs_embeds = self.transformer.wte(input_ids)

        position_embeds = self.transformer.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.transformer.drop(hidden_states)

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for i, block in enumerate(self.transformer.h):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = block(
                hidden_states,
                layer_past=past_key_values[i],
                attention_mask=attention_mask,
                head_mask=head_mask[i] if head_mask is not None else None,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[1],) # layer_outputs[1] is the new past_key_value
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[2 if use_cache else 1],)


        hidden_states = self.transformer.ln_f(hidden_states)

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + next_decoder_cache + \
                     (all_hidden_states if output_hidden_states else ()) + \
                     (all_self_attentions if output_attentions else ())
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=next_decoder_cache, # This should be a tuple of tuples (k, v) for each layer
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    # The PreTrainedModel class has its own generate method, so we remove the custom one.
    # @torch.no_grad()
    # def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    # ... (original generate method removed) ...
