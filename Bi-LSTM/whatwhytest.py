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
PATH = "./lightning_logs/version_1/checkpoints/epoch=29-step=330.ckpt"
batch_size = 64
epochs = 30
dropout = 0.4
rnn_hidden = 768
rnn_layer = 1
class_num = 4
lr = 0.001

# todo: custom dataset
import json


class MydataSet(Dataset):
    def __init__(self, path, split):
        self.data = []
        with open(path, "r", encoding="utf-8") as file:  # Use the path parameter
            for line in file:
                self.data.append(json.loads(line))

    def __getitem__(self, item):
        text = self.data[item]["msg"]  # 根据实际的键名调整
        label = 0  # 根据实际的键名调整
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
    def __init__(self, drop, hidden_dim, output_dim):
        super(BiLSTMLighting, self).__init__()
        self.model = BiLSTMClassifier(drop, hidden_dim, output_dim)  # setup model
        self.criterion = nn.CrossEntropyLoss()  # setup loss function
        self.train_dataset = MydataSet("./data/archive/train_clean.csv", "train")
        self.val_dataset = MydataSet("./data/archive/val_clean.csv", "train")
        self.test_dataset = MydataSet("./data/powertoys/test_cleanjava.csv", "train")

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=lr)
        return optimizer

    def forward(self, input_ids, attention_mask, token_type_ids):  # forward(self,x)
        return self.model(input_ids, attention_mask, token_type_ids)

    def train_dataloader(self):
        train_loader = DataLoader(
            dataset=self.train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True
        )
        return train_loader

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = batch  # x, y = batch
        y = one_hot(labels, num_classes=4)
        # convert one_hot_labels type to float
        y = y.to(dtype=torch.float)
        # forward pass
        y_hat = self.model(input_ids, attention_mask, token_type_ids)
        y_hat = y_hat.squeeze()  # squeeze [128, 1, 3] to [128,3]
        loss = self.criterion(y_hat, y)  # criterion(input, target)
        self.log(
            "train_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True
        )  # output loss to console
        return loss  # must return log for it to be useful

    def val_dataloader(self):
        val_loader = DataLoader(
            dataset=self.val_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False
        )
        return val_loader

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = batch
        y = one_hot(labels, num_classes=4)
        y = y.to(dtype=torch.float)
        # forward pass
        y_hat = self.model(input_ids, attention_mask, token_type_ids)
        y_hat = y_hat.squeeze()
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return loss

    def test_dataloader(self):
        test_loader = DataLoader(
            dataset=self.test_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False
        )
        return test_loader

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = batch
        # forward propagation
        y_hat = self.model(input_ids, attention_mask, token_type_ids)
        # prediction
        pred = torch.argmax(y_hat, dim=1)
        with open("predgo.txt", "a", encoding="utf-8") as f:
            f.write(str(pred))
        # return prediction results and original data
        return {"pred": pred, "text": [i[0] for i in batch], "label": labels}

    def on_test_epoch_end(self):
        all_predictions = []
        for output in self.trainer.callback_metrics:
            preds = output["pred"].tolist()
            texts = output["text"]
            labels = output["label"].tolist()

            for pred, text, label in zip(preds, texts, labels):
                updated_record = {"new_message1": text, "label": label, "result": pred}
                all_predictions.append(updated_record)

        # save as new JSONL file
        with open("predictions.jsonl", "w", encoding="utf-8") as file:
            for record in all_predictions:
                file.write(json.dumps(record) + "\n")


def test():
    # load previously trained optimal model parameters
    model = BiLSTMLighting.load_from_checkpoint(
        checkpoint_path=PATH, drop=dropout, hidden_dim=rnn_hidden, output_dim=class_num
    )
    trainer = Trainer(fast_dev_run=False)
    result = trainer.test(model)
    print(result)


token = BertTokenizer.from_pretrained("bert-base-uncased")
if __name__ == "__main__":

    model = BiLSTMLighting.load_from_checkpoint(
        checkpoint_path=PATH, drop=dropout, hidden_dim=rnn_hidden, output_dim=class_num
    )
    trainer = Trainer()
    trainer.test(model)
