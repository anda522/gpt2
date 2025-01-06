# 124M GPT2 模型配置
GPT_CONFIG_124M = {
    "vocab_size": 50257,  # 词表大小
    "ctx_len": 256,      # 上下文长度
    "emb_dim": 768,       # 嵌入维度
    "n_heads": 12,        # 注意力头（attention heads）的数量
    "n_layers": 12,       # 模型层数
    "drop_rate": 0.1,     # Dropout rate
    "qkv_bias": False     # Query-Key-Value bias
}

BASE_CONFIG = {
    "vocab_size": 50257,  # Vocabulary size
    "ctx_len": 1024,      # Context length
    "drop_rate": 0.0,     # Dropout rate
    "qkv_bias": True      # Query-key-value bias
}

model_configs = {
    "gpt2-small": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

# allowed model names
model_names = {
    # "gpt2-small": "openai-community/gpt2",         # 124M
    "gpt2-small": "weights/gpt2-small",         # 124M
    "gpt2-medium": "openai-community/gpt2-medium", # 355M
    "gpt2-large": "openai-community/gpt2-large",   # 774M
    "gpt2-xl": "openai-community/gpt2-xl"          # 1558M
}