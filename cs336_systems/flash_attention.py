import math

import torch
import triton
import triton.language as tl


def flash_backward(Q, K, V, O, L, dO, is_causal):
    scale = 1 / math.sqrt(Q.shape[-1])
    S = Q @ K.transpose(-1, -2) * scale
    if is_causal:
        mask = torch.arange(Q.shape[-2], device=Q.device)[:, None] >= torch.arange(K.shape[-2], device=Q.device)[None, :]
        S = torch.where(mask, S, -1e6)
    P = torch.exp(S - L[..., None])
    dV = P.transpose(-1, -2) @ dO
    dP = dO @ V.transpose(-1, -2)
    D = (O * dO).sum(dim=-1, keepdim=True)
    dS = P * (dP - D)
    dQ = dS @ K * scale
    dK = dS.transpose(-1, -2) @ Q * scale
    return dQ, dK, dV


flash_backward_compiled = torch.compile(flash_backward)


class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        B_q, B_k = 16, 16
        scale = 1 / math.sqrt(Q.shape[-1])
        O = torch.empty_like(Q)
        L = torch.empty(Q.shape[:-1], device=Q.device)
        for i in range(0, Q.shape[-2], B_q):
            Q_i = Q[..., i:i + B_q, :]
            O_i = torch.zeros_like(Q_i)
            l_i = torch.zeros(Q_i.shape[:-1], device=Q.device)
            m_i = torch.full(Q_i.shape[:-1], -torch.inf, device=Q.device)
            for j in range(0, K.shape[-2], B_k):
                K_j = K[..., j:j + B_k, :]
                V_j = V[..., j:j + B_k, :]
                S_ij = Q_i @ K_j.transpose(-1, -2) * scale
                if is_causal:
                    q_idx = torch.arange(i, i + Q_i.shape[-2], device=Q.device)
                    k_idx = torch.arange(j, j + K_j.shape[-2], device=Q.device)
                    S_ij = torch.where(q_idx[:, None] >= k_idx[None, :], S_ij, -1e6)
                m_new = torch.maximum(m_i, S_ij.max(dim=-1).values)
                P_ij = torch.exp(S_ij - m_new[..., None])
                alpha = torch.exp(m_i - m_new)
                l_i = alpha * l_i + P_ij.sum(dim=-1)
                O_i = alpha[..., None] * O_i + P_ij @ V_j
                m_i = m_new
            O[..., i:i + B_q, :] = O_i / l_i[..., None]
            L[..., i:i + B_q] = m_i + torch.log(l_i)
        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        L, Q, K, V, O = ctx.saved_tensors
        dQ, dK, dV = flash_backward_compiled(Q, K, V, O, L, dO, ctx.is_causal)
        return dQ, dK, dV, None


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    Q_i = tl.load(Q_block_ptr)
    O_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m_i = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)
    q_idx = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)

    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K_j = tl.load(K_block_ptr)
        V_j = tl.load(V_block_ptr)
        S_ij = tl.dot(Q_i, tl.trans(K_j)) * scale
        if is_causal:
            k_idx = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            S_ij += tl.where(q_idx[:, None] >= k_idx[None, :], 0, -1e6)
        m_new = tl.maximum(m_i, tl.max(S_ij, axis=1))
        P_ij = tl.exp(S_ij - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = alpha * l_i + tl.sum(P_ij, axis=1)
        O_i = tl.dot(P_ij.to(V_j.dtype), V_j, acc=alpha[:, None] * O_i)
        m_i = m_new
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    O_i = O_i / l_i[:, None]
    tl.store(O_block_ptr, O_i.to(O_block_ptr.type.element_ty))
    tl.store(L_block_ptr, m_i + tl.log(l_i))


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        B_q, B_k = 16, 16
        batch_size, N_q, D = Q.shape
        N_k = K.shape[1]
        O = torch.empty_like(Q)
        L = torch.empty((batch_size, N_q), device=Q.device)
        flash_fwd_kernel[(triton.cdiv(N_q, B_q), batch_size)](
            Q, K, V,
            O, L,
            *Q.stride(),
            *K.stride(),
            *V.stride(),
            *O.stride(),
            *L.stride(),
            N_q, N_k,
            1 / math.sqrt(D),
            D=D,
            Q_TILE_SIZE=B_q,
            K_TILE_SIZE=B_k,
            is_causal=is_causal,
        )
        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        L, Q, K, V, O = ctx.saved_tensors
        dQ, dK, dV = flash_backward_compiled(Q, K, V, O, L, dO, ctx.is_causal)
        return dQ, dK, dV, None
