from transformers import AutoModel, BertTokenizerFast
import json
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F


from transformers import AutoModel

class SAUTEPipeline:
    def __init__(self):
        model_name = "bert-base-uncased"
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.saute_model = AutoModel.from_pretrained("JustinDuc/saute", trust_remote_code=True).to(self.device)

    def inference(self, filepath):
        with open(filepath, "r") as f:
            dialogs = json.load(f)
        mask_token = self.tokenizer.mask_token
        mask_token_id = self.tokenizer.mask_token_id

        names = []
        edu_embeddings = []
        attention_mask = []


        for dialog in dialogs:
            edu = dialog["edu"]
            split_edu = edu.split(mask_token)
            names.append(dialog["speaker"])


            sentence_emb = []
            attention_emb = []
            for index, part in enumerate(split_edu):
                tokens = self.tokenizer(part.strip())

                sentence_emb.append(tokens["input_ids"])
                attention_emb.append(tokens["attention_mask"])

                if index < len(split_edu) - 1:
                    sentence_emb.append([mask_token_id])
                    attention_emb.append([1])
            edu_embeddings.append(torch.tensor([ss for s in sentence_emb for ss in s]))
            attention_mask.append(torch.tensor([ss for s in attention_emb for ss in s]))

        padded_edu_embeddings = pad_sequence(edu_embeddings, batch_first=True).unsqueeze(0).to(self.device)
        padded_attention_mask = pad_sequence(attention_mask, batch_first=True).unsqueeze(0).to(self.device)

        logits = self.saute_model(padded_edu_embeddings, [names], padded_attention_mask).logits.squeeze(0)

        temperature = 1.0
        probs = F.softmax(logits / temperature, dim=-1)

        mask_positions = (padded_edu_embeddings == mask_token_id).squeeze(0)

        output_ids = padded_edu_embeddings.clone().squeeze(0)

        for b in range(probs.shape[0]):  # batch size
            for i in range(probs.shape[1]):  # sequence length
                if mask_positions[b, i]:
                    sampled_token = torch.multinomial(probs[b, i], num_samples=1)
                    output_ids[b, i] = sampled_token
                

        dialog_output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        return [{"speaker" : names[i], "edu": dialog_output[i]} for i in range(output_ids.shape[0])]

def inference(args):
    pipeline = SAUTEPipeline()
    print(pipeline.inference(args.filepath))



