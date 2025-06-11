import torch
import torch.nn as nn
from transformers import PreTrainedModel, AutoModel
from transformers.modeling_outputs import MaskedLMOutput
from sources.saute_config import SAUTEConfig


class SAUTE(nn.Module):
    def __init__(self,
            config      : SAUTEConfig
        ):
        super().__init__()

        self.d_model = config.hidden_size
        
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads

        self.key_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.val_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.query_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self,
            input_ids           : torch.Tensor,
            speaker_names       : list[str],
            token_embeddings    : torch.Tensor,
            edu_embeddings      : torch.Tensor
        ):
        
        B, T, L = input_ids.shape

        speaker_memories = [{} for _ in range(B)]
        H = self.num_heads
        d = self.head_dim

        speaker_matrices = torch.zeros(B, T, H, d, d, device=edu_embeddings.device)

        for b in range(B):
            for t in range(T):
                speaker = speaker_names[b][t]
                e_t = edu_embeddings[b, t]  # (D)

                if speaker not in speaker_memories[b]:
                    speaker_memories[b][speaker] = {
                        'kv_sum': torch.zeros(self.num_heads, self.head_dim, self.head_dim, device=e_t.device)
                    }

                mem = speaker_memories[b][speaker]

                k_t = self.key_proj(e_t).view(self.num_heads, self.head_dim)
                v_t = self.val_proj(e_t).view(self.num_heads, self.head_dim)
                kv_t = torch.einsum("hd,he->hde", k_t, v_t)

                mem['kv_sum'] = mem['kv_sum'] + kv_t
                speaker_matrices[b, t] = mem['kv_sum']

        query_emb = self.query_proj(token_embeddings)
        query = query_emb.view(B, T, L, H, d)

        contextual = []
        for b in range(B):
            head_outputs = []
            for t in range(T):
                speaker = speaker_names[b][t]
                M = speaker_matrices[b, t]
                q = query[b, t]
                q = q.transpose(0, 1)
                a = torch.matmul(q, M)
                a = a.transpose(0, 1).contiguous().view(L, -1)
                contextual_token = token_embeddings[b, t] + a
                head_outputs.append(contextual_token)
            contextual.append(torch.stack(head_outputs))
        contextual_tokens = torch.stack(contextual)
        
        return contextual_tokens

class EDUSpeakerAwareMLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_name = "bert-base-uncased"

        self.edu_encoder = AutoModel.from_pretrained(model_name)
        for param in self.edu_encoder.parameters():
            param.requires_grad = False

        self.d_model = config.hidden_size
        self.query_proj = nn.Linear(config.hidden_size, config.hidden_size, bias = False)

        encoder_layer = nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_hidden_layers)

        self.saute = SAUTE(config)

    def forward(self, input_ids, attention_mask, speaker_names):
        """
        input_ids: (B, T, L)
        attention_mask: (B, T, L)
        speaker_names: list of list of strings, shape (B, T)
        """
        B, T, L = input_ids.shape

        with torch.no_grad():
            input_ids_flat = input_ids.view(B * T, L)
            attention_mask_flat = attention_mask.view(B * T, L)
            outputs = self.edu_encoder(input_ids=input_ids_flat, attention_mask=attention_mask_flat)
            token_embeddings = outputs.last_hidden_state

        token_embeddings = token_embeddings.view(B, T, L, self.d_model)
        edu_embeddings = token_embeddings.mean(dim=2)

        contextual_tokens = self.saute(input_ids, speaker_names, token_embeddings, edu_embeddings)

        edu_tokens = contextual_tokens.view(B * T, L, self.d_model)
        encoded_edu = self.transformer(edu_tokens)
        encoded = encoded_edu.view(B, T, L, self.d_model)

        return encoded, 0

class UtteranceEmbedings(PreTrainedModel):
    config_class = SAUTEConfig

    def __init__(self, config : SAUTEConfig):
        super().__init__(config)
        
        self.lm_head    = nn.Linear(config.hidden_size, config.vocab_size)
        self.saute_unit = EDUSpeakerAwareMLM(config)

        self.config : SAUTEConfig = config
        
        self.init_weights()

    def forward(
        self,
        input_ids       : torch.Tensor,
        speaker_names   : list[str],
        attention_mask  : torch.Tensor  = None,
        labels          : torch.Tensor  = None
    ):
        X, flop_penalty = self.saute_unit.forward(
            input_ids       =   input_ids,
            speaker_names   =   speaker_names,
            attention_mask  =   attention_mask
        )
        
        logits = self.lm_head(X)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        return MaskedLMOutput(loss=loss, logits=logits)
