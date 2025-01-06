import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6, 7"

tokenizer = tiktoken.get_encoding("gpt2")
# print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))

class SpamDataset(Dataset):
    def __init__(self, csv, tokenizer, max_length=None, pad_id=50256):
        super().__init__()
        self.data = pd.read_csv(csv)
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        if max_length is None:
            max_length = 0
            for text in self.encoded_texts:
                if len(text) > max_length:
                    max_length = len(text)
            self.max_length = max_length
        else:
            self.max_length = max_length
            self.encoded_texts = [
                text[:max_length] for text in self.encoded_texts
            ]
        
        self.encoded_texts = [
            text + [pad_id] * (self.max_length - len(text)) for text in self.encoded_texts
        ]
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return (
            torch.tensor(self.encoded_texts[index]),
            torch.tensor(self.data["Label"][index])
        )

def get_dataloader():
    train_dataset = SpamDataset(
        "./data/classify_finetune/train.csv",
        max_length=None,
        tokenizer=tokenizer
    )

    val_dataset = SpamDataset(
        "./data/classify_finetune/validation.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer
    )

    test_dataset = SpamDataset(
        "./data/classify_finetune/test.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer
    )

    num_workers = 0
    batch_size = 8

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )

    return train_loader, val_loader, test_loader

# input_batch: [8, 120] target_batch: [8]
train_loader, val_loader, test_loader = get_dataloader()

# 130 19 38
# print(len(train_loader), len(val_loader), len(test_loader))

from transformers import GPT2Model
from all_code import GPTModel, BASE_CONFIG, load_weights

weights_path = "weights/gpt2-small"
gpt_hf = GPT2Model.from_pretrained(weights_path)
gpt_hf.eval()

model = GPTModel(BASE_CONFIG)
load_weights(model, gpt_hf)
model.eval()

for param in model.parameters():
    param.requires_grad = False

num_classes = 2
model.out_head = torch.nn.Linear(BASE_CONFIG["emb_dim"], num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True

for param in model.final_norm.parameters():
    param.requires_grad = True

# inputs = tokenizer.encode("Do you have time?")
# inputs = torch.tensor(inputs).unsqueeze(0)
# print(inputs)
# print(inputs.shape)

# with torch.no_grad():
#     outputs = model(inputs)
# print("Last output token:", outputs[:, -1, :])

def calc_accuracy_loader(loader, model, device, num_batches=None):
    model.eval()
    correct, total = 0, 0

    if num_batches is None:
        num_batches = len(loader)
    else:
        num_batches = min(num_batches, len(loader))
    
    for i, (input_batch, target_batch) in enumerate(loader):
        if i >= num_batches:
            break
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        with torch.no_grad():
            logits = model(input_batch)[:, -1, :]
        predicted_labels = torch.argmax(logits, dim=-1)
        total += predicted_labels.shape[0]
        correct += (predicted_labels == target_batch).sum().item()
    return correct / total

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss

def calc_loss_loader(loader, model, device, num_batches=None):
    total_loss = 0.
    if len(loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(loader)
    else:
        num_batches = min(num_batches, len(loader))
    for i, (input_batch, target_batch) in enumerate(loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()
    return total_loss / num_batches

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def train_classifier(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter, tokenizer):
    # 初始化列表以跟踪损失和看到的示例
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    # 主要的训练循环
    for epoch in range(num_epochs):
        model.train()  # 将模型设置为训练模式
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # 重置上一个 epoch 的损失梯度
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # 计算损失梯度
            optimizer.step() # 使用损失梯度更新模型权重
            examples_seen += input_batch.shape[0] # 新功能：跟踪示例而不是标记
            global_step += 1

            # 可选的评估步骤
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Epoch {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # 计算每个 epoch 后的准确率
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen

import time

start_time = time.time()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

num_epochs = 5
train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=50, eval_iter=5,
    tokenizer=tokenizer
)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")