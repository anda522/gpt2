import torch

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # 增加batch维度
    return encoded_tensor.cuda()

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # 去掉batch维度
    return tokenizer.decode(flat.tolist())

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)

    logits = model(input_batch)
    logits = logits.flatten(0, 1)
    loss = torch.nn.functional.cross_entropy(logits, target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None): # num_batches为计算损失的批次范围
    total_loss = 0.
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        # 取num_batches和len(data_loader)两者较小值以匹配data_loader中的总批次数量
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size, top_k=10, temperature=1.5
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))  # 简洁的打印格式
    model.train()

def generate(model, idx, max_new_tokens, context_size, temperature, top_k=None):
    # 循环与之前相同：获取logits，并仅关注最后一步。
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        # 使用top_k采样对logits值进行过滤
        if top_k is not None:
            # 仅保留top_k的值
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)
        # 使用温度缩放
        if temperature > 0.0:
            logits = logits / temperature
            # 使用softmax函数得到概率
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)
            # 从概率分布中采样
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
        # 否则和之前的generate_simple函数中的处理相同，使用argmax函数取得概率最大的token
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)
        # 和之前相同的序列拼接处理
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx