import argparse
from pickle import NONE
from tkinter import NO
import torch
from datasets import load_dataset  # hugging-face dataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import torch.optim as optim
from torch.nn.functional import one_hot
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import os

torch.backends.cudnn.enable = True
torch.backends.cudnn.benchmark = True
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.set_float32_matmul_precision("high")
PATH = "./lightning_logs/best-checkpoint-epoch=05-val_loss=0.65.ckpt"
batch_size = 64
epochs = 30
dropout = 0.4
rnn_hidden = 768
rnn_layer = 1
class_num = 4
lr = 0.001
global repo

# todo: custom dataset
import json


class MydataSet(Dataset):
    def __init__(self, path, split):
        self.data = []
        if not os.path.exists(path):
            print(f"❌ File not found: {path}")
            return

        with open(path, "r", encoding="utf-8") as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data_item = json.loads(line)
                    self.data.append(data_item)
                except json.JSONDecodeError as e:
                    print(f"⚠️ Skipping line {line_num}: {e}")

        print(f"✅ Loaded {len(self.data)} samples from {path}")

    def __getitem__(self, item):
        if item >= len(self.data):
            return "", 0
        text = self.data[item].get("msg", "")
        label = self.data[item].get("label", 0)
        return text, label

    def __len__(self):
        return len(self.data)


# todo: define batch processing function
def collate_fn(data):
    sents = [i[0] for i in data]
    labels = [i[1] for i in data]
    # tokenize and encode
    data = token.batch_encode_plus(
        batch_text_or_text_pairs=sents,  # single sentence encoding
        truncation=True,  # truncate when sentence length exceeds max_length
        padding="max_length",  # pad all to max_length
        max_length=200,
        return_tensors="pt",  # return in pytorch format, can be tf,pt,np, default returns list
        return_length=True,
    )

    # input_ids: encoded numbers
    # attention_mask: padded positions are 0, other positions are 1
    input_ids = data["input_ids"]  # input_ids are the encoded words
    attention_mask = data["attention_mask"]  # pad positions are 0, other positions are 1
    token_type_ids = data[
        "token_type_ids"
    ]  # (for sentence pairs) first sentence and special tokens are 0, second sentence is 1
    labels = torch.LongTensor(labels)  # labels for this batch

    # print(data['length'], data['length'].max())
    return input_ids, attention_mask, token_type_ids, labels


# todo: define model, upstream uses bert pretrained, downstream task uses bidirectional LSTM model, finally add a fully connected layer
class BiLSTMClassifier(nn.Module):
    def __init__(self, drop, hidden_dim, output_dim):
        super(BiLSTMClassifier, self).__init__()
        self.drop = drop
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # load bert model, generate embedding layer
        self.embedding = BertModel.from_pretrained("bert-base-uncased")
        # remove move to gpu
        # freeze upstream model parameters (do not learn pretrained model parameters)
        for param in self.embedding.parameters():
            param.requires_grad_(False)
        # generate downstream RNN layer and fully connected layer
        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=self.hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=self.drop,
        )
        self.fc = nn.Linear(self.hidden_dim * 2, self.output_dim)
        # When using CrossEntropyLoss as loss function, no activation needed. Because CrossEntropyLoss actually implements softmax-log-NLLLoss together.

    def forward(self, input_ids, attention_mask, token_type_ids):
        embedded = self.embedding(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        embedded = (
            embedded.last_hidden_state
        )  # dimension 0 is the embedding we need, embedding.last_hidden_state = embedding[0]
        out, (h_n, c_n) = self.lstm(embedded)
        output = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        output = self.fc(output)
        return output


# todo: define pytorch lightning
class BiLSTMLighting(pl.LightningModule):
    def __init__(self, drop, hidden_dim, output_dim, test_data_path=None):
        super(BiLSTMLighting, self).__init__()
        self.model = BiLSTMClassifier(drop, hidden_dim, output_dim)  # setup model
        self.criterion = nn.CrossEntropyLoss()  # setup loss function

        # ✅ Store test step outputs manually
        self.test_step_outputs = []

        # Only initialize test dataset for testing
        self.test_dataset = None
        if test_data_path:
            self.test_dataset = MydataSet(test_data_path, "train")

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=lr)
        return optimizer

    def forward(self, input_ids, attention_mask, token_type_ids):  # forward(self,x)
        return self.model(input_ids, attention_mask, token_type_ids)

    def test_dataloader(self):
        if self.test_dataset is None or len(self.test_dataset) == 0:
            print("❌ No test dataset available!")
            return None

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=4,  # ✅ Add workers as suggested
        )
        return test_loader

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = batch

        # forward propagation
        y_hat = self.model(input_ids, attention_mask, token_type_ids)
        # prediction
        pred = torch.argmax(y_hat, dim=1)

        # Write raw predictions to file
        path = f"./data/Powertoys/predgo.txt"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(str(pred) + "\n")

        # ✅ Get original texts from dataset using batch indices
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + len(labels), len(self.test_dataset))
        texts = [self.test_dataset.data[i].get("msg", "") for i in range(batch_start, batch_end)]
        shas = [self.test_dataset.data[i].get("sha", "") for i in range(batch_start, batch_end)]

        # ✅ Store outputs for later processing
        output = {
            "pred": pred.cpu(),  # Move to CPU to avoid memory issues
            "text": texts,
            "label": labels.cpu(),
            "sha": shas,
        }
        self.test_step_outputs.append(output)

        return output

    def on_test_epoch_end(self):
        if not self.test_step_outputs:
            print("❌ No test outputs to process!")
            return

        all_predictions = []

        print(f"📊 Processing {len(self.test_step_outputs)} batches...")

        # ✅ Process collected outputs from test steps
        for batch_num, output in enumerate(self.test_step_outputs):
            preds = output["pred"].tolist()
            texts = output["text"]
            labels = output["label"].tolist()
            shas = output["sha"]

            print(f"Batch {batch_num}: {len(preds)} predictions, {len(texts)} texts, {len(labels)} labels")

            # Make sure all lists have the same length
            min_len = min(len(preds), len(texts), len(labels))

            for i in range(min_len):
                updated_record = {
                    "sha": shas[i],
                    "new_message1": texts[i],
                    "label": labels[i],
                    "predicted_result": preds[i],
                }
                all_predictions.append(updated_record)

        # Save as new JSONL file
        output_path = "./data/Powertoys/predictions.jsonl"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as file:
            for record in all_predictions:
                file.write(json.dumps(record) + "\n")

        print(f"✅ Saved {len(all_predictions)} predictions to {output_path}")

        # Calculate some basic stats
        if all_predictions:
            label_counts = {}
            result_counts = {}

            for pred in all_predictions:
                label = pred["label"]
                result = pred["predicted_result"]

                label_counts[label] = label_counts.get(label, 0) + 1
                result_counts[result] = result_counts.get(result, 0) + 1

            print(f"📊 Label distribution: {label_counts}")
            print(f"📊 Prediction distribution: {result_counts}")

        # ✅ Clear outputs for next test run
        self.test_step_outputs.clear()


token = BertTokenizer.from_pretrained("bert-base-uncased")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Args to run the filter script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("-t", "--test_data", required=True, help="test file path")

    args = parser.parse_args()
    test_data_path = args.test_data

    print(f"🔄 Loading model from {PATH}")
    print(f"📁 Test data: {test_data_path}")

    model = BiLSTMLighting.load_from_checkpoint(
        checkpoint_path=PATH,
        drop=dropout,
        hidden_dim=rnn_hidden,
        output_dim=class_num,
        test_data_path=test_data_path,
    )

    print(f"📊 Dataset size: {len(model.test_dataset) if model.test_dataset else 0}")

    trainer = Trainer()
    trainer.test(model)

    print("🎉 Testing completed!")
