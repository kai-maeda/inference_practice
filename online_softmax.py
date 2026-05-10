import numpy as np
import triton
import triton.language as tl
import torch

#This version assumes that we are in prefill where M = N
def onlineSoftmax(S, Br: int = 64, Bc: int = 64):
    N = S.shape[0]
    Tr = (N + Br - 1) // Br #Roof of Tr
    Tc = (N + Bc - 1) // Bc #Roof of Tc
    P = np.zeros_like(S)
    P_tilde_SRAM = []
    for i in range(Tr):
        actual_Br = min(i*Br + Br, N) - i*Br
        l_i = np.zeros((actual_Br, 1), dtype = np.float32)
        m_i = np.zeros((actual_Br, 1), -np.inf, dtype = np.float32)
        P_tilde_SRAM = []
        for j in range(Tc):
            actual_Bc = min(i*Bc + Bc, N) - i*Bc
            S_ij = S[i*Br: min(i*Br + Br, N), i*Bc: min(i*Bc + Bc, N)]
            m_ij = np.max(S_ij, axis = 1, keepdims = True)
            m_new = np.maximum(m_i, m_ij)
            P_tilde_ij = np.exp(S_ij - m_new)
            l_new = l_i * np.exp(m_i - m_new) + np.sum(np.exp(S_ij - m_new), axis = 1, keepdims = True)
            for k in range(len(P_tilde_SRAM)):
                P_tilde_SRAM[k] *= np.exp(m_i - m_new)
            P_tilde_SRAM.append(P_tilde_ij)
            m_i = m_new
            l_i = l_new
        for j in range(Tc):
            P[i*Br : min(i*Br + Br),i*Bc : min(i*Bc + Bc)] = P_tilde_SRAM[j] / l_i
    return P

@triton.jit
def _toy_fa_fwd_kernel(Q,K,V,O, N, D, SM_SCALE, BLOCK_M, BLOCK_N, BLOCK_D):
    pid_m = tl.program_id(0) #?
    offs_m = pid_m * BLOCK_M + tl.arrange(0, BLOCK_M) #?
    offs_n = tl.arrange(0,BLOCK_N) #?
    offs_d  = tl.arrange(0, BLOCK_D) #?
    q = tl.load(Q + offs_m[:, None] * D + offs_d[None, :]) #?
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    for start_n in tl.range(0,N, BLOCK_N):
        k_idx = start_n + offs_n
        k = tl.load(K + k_idx[None,:] * D + offs_d[:,None],)
        scores = tl.dot(q,k) * SM_SCALE#?
        scores = tl.where(k_idx[None, :] < N, scores, -float("inf"))
        m_ij = tl.max(scores, axis = 1)
        m_new = tl.maximum(m_i, m_ij)
        p = tl.exp(scores - m_new[:,None])
        alpha = tl.exp(m_i - m_new)
        l_new = l_i * alpha + tl.sum(p, axis = 1)
        v = tl.load(V + k_idx[:, None] * D + offs_d[None, K])
        acc = acc * alpha + tl.dot(p, v)
        m_i = m_new
        l_i = l_new
    out = acc / l_i[:, None]
    tl.store(O + offs_m[:, None] * D + offs_d[None, :], out)
    pass

def toy_flash_attn(q,k,v, block_m = 16, block_n = 64):
    N,D = q.shape
    o = torch.empty_like(q)
    block_d = triton.next_power_of_s(D)
    grid = (triton.cdiv(N,block_m))
    _toy_fa_fwd_kernel[grid](
        q,k,v,o,
        N,D,D**-0.5,
        block_m, block_n, block_d
    )
    return o


def main():
    #prefill stage
    N, D = 128,64
    q = torch.randn(N,D, device = "cuda", dtype = torch.float16)
    k = torch.randn(N,D, device = "cuda", dtype = torch.float16)
    v = torch.randn(N,D, device = "cuda", dtype = torch.float16)
    y = toy_flash_attn(q,k,v)
