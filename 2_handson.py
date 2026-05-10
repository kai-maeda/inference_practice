#* Hands-on: implement multi-head attention from scratch in PyTorch in <50 lines. No reference. Then implement KV-cached generation.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.h_d = d_model // h
        self.qkv_proj = nn.Linear(d_model, 3*d_model)
        self.o_proj = nn.Linear(d_model, d_model) 

    def forward(self, x):
        B,T,C = x.size() #Batch, Sequence, d_model

        #1. Fused projection and split into Q,K,V
        q,k,v = self.qkv_proj(x).split(self.d_model, dim = 1)

        #2. Reshape and transpose for multihead (B, h, T, h_d)
        q = q.view(B,T,self.h,self.h_d).transpose(1,2)
        k = k.view(B,T,self.h,self.h_d).transpose(1,2)
        v = v.view(B,T,self.h,self.h_d).transpose(1,2)

        #3. create causal mask
        mask = torch.tril(torch.ones(T,T)).view(1,1,T,T).to(x.device)

        #4. scaled dot product attention
        scores = (q @ k.transpose(-2,-1)) / math.sqrt(self.h_d)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim = -1)

        #5. Multiply with values and reshape back
        y = (attn @ v).transpose(1,2).contiguous().view(B,T,C)

        return self.o_proj(y)
    
class KVCachedAttention(nn.Module):
    def __init__(self,d_model, h):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.d_h = d_model // h
        self.qkv_proj = nn.Linear(d_model, 3* d_model)
        self.o_proj = nn.Linear(d_model, d_model)

    def forward(self, x, kv_cache = None):
        B,T,C = x.size()
        q,k,v = self.qkv_proj(x).split(self.d_model, dim = -1)
        q = q.view(B, T, self.h, self.d_h).transpose(1,2)
        k = k.view(B, T, self.h, self.d_h).transpose(1,2)
        v = v.view(B, T, self.h, self.d_h).transpose(1,2)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim = 2)
            v = torch.cat([past_v, v], dim = 2)

        new_kv_cache = (k,v)
        #q: (B, h, 1, d_h)
        #k: (B, h, T, d_h)
        #v: (B, h, T, d_h)
        score = (q @ k.transpose(-2,-1)) / math.sqrt(self.d_h)
        #attn: (B, h, 1, T)
        attn = F.softmax(score, dim = -1)
        #out: (B, h, 1, d_h)
        out = (attn @ v).transpose(1,2).contiguous().view(B,T,C)
        return self.o_proj(out), kv_cache
    #concatenating kv cache because GPU allocates a brand new, contiguous block of HBM
    #PagedAttention allocates KV cache in non-contiguous OS-style blocks