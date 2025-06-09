from datasets import load_dataset
from transformers import BertTokenizerFast
from functools import reduce
import torch
import random
import json

MAX_EDU_LEN = 128
MAX_SPEAKER_NAME_LEN = 5
MAX_EDUS_PER_DIALOG = 100

class SAUTEDataset(torch.utils.data.Dataset):
    
    dialog_formats = [
        "edu",
        "full"
    ]
    
    def __init__(
        self,
        split         : str = "train",
        dialog_format : str = "edu",
        dataset       : str = "allenai/soda"
    ):
        assert dialog_format in SAUTEDataset.dialog_formats, f"Unknown dialog format {dialog_format}. Available dialog formats are {str(SAUTEDataset.dialog_formats)}"
        
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.dataset = load_dataset(dataset, split=split)
        
        self.dialog_format = dialog_format

    def __len__(self):
        return len(self.dataset)

    def mask_tokens(
        self,
        inputs : torch.Tensor
    ):
        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, 0.15)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        return inputs, labels

    def __getitem__(
        self,
        idx
    ):
        item = self.dataset[idx]
        edus = (item['dialogue'])[:MAX_EDUS_PER_DIALOG]
        speakers = (item['speakers'])[:MAX_EDUS_PER_DIALOG]
        # print(list(zip(edus, speakers)))
        # if self.dialog_format == "full":
        #     edus = ["\n".join(map(lambda x : "[" + x[0] + "]: " + x[1], zip(speakers, edus)))]
        #     # print(edus)

        # print(edus)
        tokenized = self.tokenizer(edus, padding="max_length", truncation=True,
                                   max_length=MAX_EDU_LEN, return_tensors="pt", add_special_tokens = False)
    
        if self.dialog_format == "full":
            unique_speakers     = list(set(speakers))
            unique_speaker_ids  = self.tokenizer(unique_speakers, padding="max_length", truncation=True, max_length = MAX_SPEAKER_NAME_LEN, return_tensors="pt", add_special_tokens=False)
            all_speaker_ids     = torch.stack([unique_speaker_ids["input_ids"][unique_speakers.index(s)] for s in speakers])
            all_attention_masks = torch.stack([unique_speaker_ids["attention_mask"][unique_speakers.index(s)] for s in speakers])
    
        # print(tokenized)
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        input_ids, labels = self.mask_tokens(input_ids)
        
        if self.dialog_format == "full":
            if (all_speaker_ids.shape[0] == input_ids.shape[0]):
                # print(len(speakers))
                # print(len(edus))
                # print(edus)
                # print(speakers)
                # print("Huh?", all_speaker_ids.shape, input_ids.shape)
                input_ids = torch.concat([all_speaker_ids, input_ids], dim=1)
                attention_mask = torch.concat([all_attention_masks, attention_mask], dim=1)
                labels = torch.concat([torch.full((input_ids.shape[0], MAX_SPEAKER_NAME_LEN), -100), labels], dim=1)

        start_tokens = torch.full((input_ids.shape[0], 1), self.tokenizer.cls_token_id)
        end_tokens = torch.full((input_ids.shape[0], 1), self.tokenizer.sep_token_id)
        input_ids = torch.concat([start_tokens, input_ids, end_tokens], dim=1)
        attention_mask = torch.concat([torch.ones(input_ids.shape[0], 1), attention_mask, torch.ones(input_ids.shape[0], 1)], dim=1)
        labels = torch.concat([torch.full((input_ids.shape[0], 1), -100), labels, torch.full((input_ids.shape[0], 1), -100)], dim=1)

        speaker_names = [s if s else "unknown" for s in speakers]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            **({"speaker_names": speaker_names} if self.dialog_format == "edu" else {}),
            "labels": labels
        }
