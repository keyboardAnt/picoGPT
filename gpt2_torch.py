import torch
import torch.nn.functional as F
from torch import Tensor

from utils import cast_to_torch_recursively

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(torch.sqrt(2 / torch.pi) * (x + 0.044715 * x**3)))


def softmax(x):
    return F.softmax(x, dim=-1)


def layer_norm(x, g, b, eps: float = 1e-5):
    mean = torch.mean(x, dim=-1, keepdim=True)
    variance = torch.var(x, dim=-1, keepdim=True)
    x = (x - mean) / torch.sqrt(variance + eps)
    return g * x + b


def linear(x, w, b):
    return torch.matmul(x, w) + b


def ffn(x, c_fc, c_proj):
    a = gelu(linear(x, **c_fc))
    x = linear(a, **c_proj)
    return x


def attention(q, k, v, mask):
    return softmax(torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(q.size(-1)) + mask) @ v


def mha(x, c_attn, c_proj, n_head):
    x = linear(x, **c_attn)
    qkv = torch.chunk(x, 3, dim=-1)
    qkv_heads = list(map(lambda x: torch.chunk(x, n_head, dim=-1), qkv))
    causal_mask = (1 - torch.triu(torch.ones_like(x[:, :, 0], dtype=x.dtype))) * -1e10
    out_heads = [attention(q, k, v, causal_mask) for q, k, v in zip(*qkv_heads)]
    x = torch.cat(out_heads, dim=-1)
    x = linear(x, **c_proj)
    return x


def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):
    x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)
    x = x + ffn(layer_norm(x, **ln_2), **mlp)
    return x


def gpt2(inputs: Tensor, wte, wpe, blocks, ln_f, n_head):
    # x = wte[inputs] + wpe[torch.arange(inputs.size(0))]
    # import pdb; pdb.set_trace()
    inputs = torch.tensor(inputs, dtype=torch.long)
    wte = Tensor(wte)
    wpe = Tensor(wpe)
    blocks = cast_to_torch_recursively(blocks)
    x = wte[inputs] + wpe[torch.arange(len(inputs))]
    for block in blocks:
        x = transformer_block(x, **block, n_head=n_head)
    x = layer_norm(x, **ln_f)
    return torch.matmul(x, wte.transpose(0, 1))


def generate(inputs, params, n_head, n_tokens_to_generate):
    from tqdm import tqdm

    inputs = torch.tensor(inputs, dtype=torch.long)
    for _ in tqdm(range(n_tokens_to_generate), "generating"):
        logits = gpt2(inputs, **params, n_head=n_head)
        next_id = torch.argmax(logits[-1]).item()
        inputs = torch.cat([inputs, torch.tensor([next_id], dtype=torch.long)])
    
    # TODO: Consider supporting a tensor
    return inputs[len(inputs) - n_tokens_to_generate:].tolist()
