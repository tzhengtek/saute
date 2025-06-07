import torch
import torch.nn as nn
from transformers import PreTrainedModel, AutoModel
from transformers.modeling_outputs import MaskedLMOutput
from saute_config import SAUTEConfig

class EDUSpeakerAwareMLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        # model_name="sentence-transformers/all-MiniLM-L6-v2"
        model_name = "bert-base-uncased"

        self.edu_encoder = AutoModel.from_pretrained(model_name)
        for param in self.edu_encoder.parameters():
            param.requires_grad = False  # frozen encoder

        self.d_model = config.hidden_size
        self.query_proj = nn.Linear(config.hidden_size, config.hidden_size, bias = False)

        encoder_layer = nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_hidden_layers)

        self.speaker_memory = {}  # Will be filled per batch
        self.lm_head = nn.Linear(config.hidden_size, self.edu_encoder.config.vocab_size)
        self.saute = SAUTE(config)

    def forward(self, input_ids, attention_mask, speaker_names):
        """
        input_ids: (B, T, L)
        attention_mask: (B, T, L)
        speaker_names: list of list of strings, shape (B, T)
        """
        B, T, L = input_ids.shape

        # Encode EDUs using frozen encoder
        with torch.no_grad():
            input_ids_flat = input_ids.view(B * T, L)
            attention_mask_flat = attention_mask.view(B * T, L)
            outputs = self.edu_encoder(input_ids=input_ids_flat, attention_mask=attention_mask_flat)
            token_embeddings = outputs.last_hidden_state  # (B*T, L, D)

        token_embeddings = token_embeddings.view(B, T, L, self.d_model)
        edu_embeddings = token_embeddings.mean(dim=2)  # (B, T, D)
        token_embeddings = self.query_proj(token_embeddings)

        contextual_tokens = self.saute(input_ids, speaker_names, token_embeddings, edu_embeddings)

        # === NEW: EDU-level Transformer ===
        edu_tokens = contextual_tokens.view(B * T, L, self.d_model)  # (B*T, L, D)
        encoded_edu = self.transformer(edu_tokens)  # (B*T, L, D)
        encoded = encoded_edu.view(B, T, L, self.d_model)  # (B, T, L, D)

        return encoded, 0

class SAUTE(nn.Module):
    def __init__(self,
            config      : SAUTEConfig
        ):
        super().__init__()

        self.d_model = config.hidden_size
        self.key_proj = nn.Linear(config.hidden_size, config.hidden_size, bias = False)
        self.val_proj = nn.Linear(config.hidden_size, config.hidden_size, bias = False)

    def forward(self,
            input_ids           : torch.Tensor,
            speaker_names       : list[str],
            token_embeddings    : torch.Tensor,
            edu_embeddings      : torch.Tensor
        ):

        # Speaker-aware memory
        B, T, L = input_ids.shape

        speaker_memories = [{} for _ in range(B)]
        speaker_matrices = torch.zeros(B, T, self.d_model, self.d_model, device=edu_embeddings.device)

        for b in range(B):
            for t in range(T):
                speaker = speaker_names[b][t]
                e_t = edu_embeddings[b, t]  # (D)

                if speaker not in speaker_memories[b]:
                    speaker_memories[b][speaker] = {
                        'kv_sum': torch.zeros(self.d_model, self.d_model, device=e_t.device),
                        # 'k_sum': torch.zeros(self.d_model, device=e_t.device),
                    }

                mem = speaker_memories[b][speaker]
                k_t = self.key_proj(e_t)
                v_t = self.val_proj(e_t)
                kv_t = torch.outer(k_t, v_t)

                # with torch.no_grad():
                mem['kv_sum'] = mem['kv_sum'] + kv_t
                # mem['k_sum'] = mem['k_sum'] + k_t

                # z = torch.clamp(mem['k_sum'] @ k_t, min=1e-6)
                # M_s = mem['kv_sum'] / z  # (D, D)

                # speaker_matrices[b, t] = M_s
                speaker_matrices[b, t] = mem['kv_sum']

        # Apply speaker matrix to each token
        speaker_matrices_exp = speaker_matrices.unsqueeze(2)  # (B, T, 1, D, D)
        token_embeddings_exp = token_embeddings.unsqueeze(-1)  # (B, T, L, D, 1)
        contextual_tokens = token_embeddings + torch.matmul(speaker_matrices_exp, token_embeddings_exp).squeeze(-1)  # (B, T, L, D)

        return contextual_tokens

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
        # print(input_ids.shape)
        X, flop_penalty = self.saute_unit.forward(
            input_ids       =   input_ids,
            speaker_names   =   speaker_names,
            attention_mask  =   attention_mask,
            # hidden_state    =   None
        )
        
        logits = self.lm_head(X)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1)) + 1e-3 * flop_penalty
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        return MaskedLMOutput(loss=loss, logits=logits)
