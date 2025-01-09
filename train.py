import torch
import urllib
import os
from dataset import create_dataloader_v1
from config import GPT_CONFIG_124M
from model import GPTModel
from tools import calc_loss_batch, evaluate_model, generate_and_print_sample

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4, 7"

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context):
    # 初始化列表以跟踪损失和已观察到的token
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # 主要的训练步骤
    for epoch in range(num_epochs):
        model.train()  # 将模型设置为训练模式
        
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # 每个epoch开始之前重新设置梯度
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # 计算损失梯度
            optimizer.step() # 利用损失梯度更新模型参数
            tokens_seen += input_batch.numel()
            global_step += 1

            # 可选的验证评估步骤
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Epoch {epoch + 1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # 在每个epoch完成后打印一个生成的文本示例
        generate_and_print_sample(
            model, train_loader.dataset.tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen

file_path = "data/the-verdict.txt"
url = "https://github.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

if not os.path.exists(file_path):
    with urllib.request.urlopen(url) as response:
        text_data = response.read().decode('utf-8')
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text_data)
else:
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

# 训练集/验证集数据比
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

# torch.manual_seed(123)
train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["ctx_len"],
    stride=GPT_CONFIG_124M["ctx_len"],
    drop_last=True,
    shuffle=True
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["ctx_len"],
    stride=GPT_CONFIG_124M["ctx_len"],
    drop_last=False,
    shuffle=False
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# checkpoint = torch.load("model_and_optimizer.pth")

model = GPTModel(GPT_CONFIG_124M)
# model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
# optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
model.train()

epoches = 20
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=epoches, eval_freq=5, eval_iter=5,
    start_context="What's wrong with you",
)
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    }, 
    "checkpoints/model_and_optimizer.pth"
)

import matplotlib.pyplot as plt

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots()

    # 绘制随着训练进行（epoch值增大）训练集损失和验证集损失的变化情况
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # 创建第二个x轴用于显示可观察的tokens
    ax2 = ax1.twiny()  # 创建一个共享相同y轴的第二个x轴
    ax2.plot(tokens_seen, train_losses, alpha=0)  # 用于对齐刻度的不可见图表
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # 调整布局以节省空间
    plt.savefig("img/loss-plot.png")
    plt.show()

epochs_tensor = torch.linspace(0, epoches, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)