import argparse
import random
from torch.utils.data import Subset
import torch
from transformers         import Trainer, TrainingArguments, BertTokenizerFast
from model  import UtteranceEmbedings
from saute_config import SAUTEConfig
from saute_datasets     import SAUTEDataset

def get_args():
    parser = argparse.ArgumentParser(description="SAUTE.")
    subparsers = parser.add_subparsers()

    # Training parameters
    train_parser = subparsers.add_parser('train', help="Train the model")
    train_parser.add_argument("--epochs", type=int, default=1,
                              help="Number of epochs")
    train_parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'gelu'],
                              help='Activation function')
    train_parser.add_argument('--layers', type=int, default=1,
                              help='Number of layers')

    # Inference
    inference_parser = subparsers.add_parser('inference', help="Inference the model")

    parser.add_argument('--datasets', type=str, default="allenai/soda",
                        help='Dataset Reposity HugginFace')
    
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'],
                        help='Device to train on')

    return parser.parse_args()

def compute_masked_accuracy(logits, labels):
    # logits: [batch_size, seq_len, vocab_size]
    # labels: [batch_size, seq_len]

    preds = torch.argmax(logits, dim=-1)  # [batch_size, seq_len]

    # Only consider masked positions (labels != -100)
    mask = labels != -100

    # Count correct predictions
    correct = (preds == labels) & mask
    accuracy = correct.sum().float() / mask.sum()

    return accuracy.item()

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    logits = torch.tensor(logits)
    labels = torch.tensor(labels)
    acc = compute_masked_accuracy(logits, labels)
    return {"masked_accuracy": acc}

def batched_saute_collator(batch):

    pad_token = "[PAD]"

    input_ids_list     = [torch.tensor(x["input_ids"]) for x in batch]
    attention_masks    = [torch.tensor(x["attention_mask"]) for x in batch]
    speaker_names_list = [x["speaker_names"] for x in batch]
    labels_list        = [torch.tensor(x["labels"]) for x in batch]

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    max_T = max(len(x) for x in speaker_names_list)
    max_L = max(seq.shape[1] for seq in input_ids_list)

    def pad_dialog(tensor, max_T, max_L, pad_val):
        return torch.nn.functional.pad(tensor, (0, max_L - tensor.shape[1], 0, max_T - tensor.shape[0]), value=pad_val)

    input_ids = torch.stack([pad_dialog(seq, max_T, max_L, tokenizer.pad_token_id) for seq in input_ids_list])
    attention_mask = torch.stack([pad_dialog(mask, max_T, max_L, 0) for mask in attention_masks])
    labels = torch.stack([pad_dialog(lbl, max_T, max_L, -100) for lbl in labels_list])  # -100: ignore index for loss

    # print(len(input_ids_list))
    if len(input_ids_list) == 1:
      input_ids_list = [input_ids_list[0].unsqueeze(0)]
      attention_masks = [attention_masks[0].unsqueeze(0)]
      labels_list = [labels_list[0].unsqueeze(0)]


    speaker_names = [
        names + [pad_token] * (max_T - len(names))
        for names in speaker_names_list
    ]

    return {
        "input_ids": input_ids,               # (B, T, L)
        "attention_mask": attention_mask,     # (B, T, L)
        "speaker_names": speaker_names,       # (B, T)
        "labels": labels                      # (B, T, L)
    }

def main():
    args = get_args()

    # Load Dataset
    train_dataset = SAUTEDataset(split="train", dialog_format="edu", dataset=args.datasets)
    eval_dataset = SAUTEDataset(split="test", dialog_format="edu", dataset=args.datasets)
    print(f"Dataset {args.datasets} is loaded !")

    # Create Subset dataset
    subset_size = 10
    indices = random.sample(range(len(eval_dataset)), subset_size)
    test_dataset = Subset(eval_dataset, indices)

    # Load Model
    model_config = SAUTEConfig(
        num_attention_heads = 12,
        num_hidden_layers   = 6,
        num_token_layers=3,
    )
    model = UtteranceEmbedings(model_config).to(args.device)
    print(f"Model is loaded on {args.device} device")

    ## Fixed 
    fixed_batch = batched_saute_collator([train_dataset[0]])
    print(fixed_batch["input_ids"].shape)

    # Training
    torch.autograd.set_detect_anomaly(True)

    print("Start Training...")
    training_args = TrainingArguments(
        output_dir="./checkpoints",
        push_to_hub=True,
        hub_strategy="end",
        hub_model_id="username/model-name",
        save_steps=5000,
        save_strategy="steps",
        eval_steps=150,
        eval_strategy="steps",
        per_device_train_batch_size=5,
        per_device_eval_batch_size=5,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=150,
        fp16=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=None,
        data_collator = batched_saute_collator,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    return

if __name__ == "__main__":
    main()