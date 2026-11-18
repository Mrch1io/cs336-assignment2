import torch
import torch.distributed as dist

from cs336_basics.model import Embedding, Linear


class DDP(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        self.handles = []
        for p in module.parameters():
            dist.broadcast(p.data, src=0)
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(self._all_reduce_grad)

    def _all_reduce_grad(self, p):
        p.grad.div_(self.world_size)
        self.handles.append(dist.all_reduce(p.grad, async_op=True))

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()


class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls, **kwargs):
        self.optimizer_cls = optimizer_cls
        self.kwargs = kwargs
        self.optimizer = None
        self.param_owner = {}
        super().__init__(params, kwargs)

    def add_param_group(self, param_group):
        super().add_param_group(param_group)
        local_params = []
        for p in param_group["params"]:
            self.param_owner[p] = len(self.param_owner) % dist.get_world_size()
            if self.param_owner[p] == dist.get_rank():
                local_params.append(p)
        if not local_params:
            return
        local_group = {**param_group, "params": local_params}
        if self.optimizer is None:
            self.optimizer = self.optimizer_cls([local_group], **self.kwargs)
        else:
            self.optimizer.add_param_group(local_group)

    def step(self, closure=None, **kwargs):
        loss = self.optimizer.step(closure, **kwargs)
        for p, owner in self.param_owner.items():
            dist.broadcast(p.data, src=owner)
        return loss


class FSDP(torch.nn.Module):
    def __init__(self, module, compute_dtype=None):
        super().__init__()
        self.module = module
        self.compute_dtype = compute_dtype
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.handles = []
        for p in module.parameters():
            dist.broadcast(p.data, src=0)
        sharded = set()
        for m in module.modules():
            if isinstance(m, (Linear, Embedding)):
                m.full_shape = m.weight.shape
                m.weight.data = m.weight.data.flatten().chunk(self.world_size)[self.rank].clone()
                m.register_forward_pre_hook(self._gather_weight)
                m.register_forward_hook(self._release_weight)
                m.register_full_backward_pre_hook(self._gather_weight)
                m.weight.register_post_accumulate_grad_hook(self._make_reduce_scatter_hook(m))
                sharded.add(m.weight)
        for p in module.parameters():
            if p not in sharded:
                p.register_post_accumulate_grad_hook(self._all_reduce_grad)

    def _all_gather(self, m, shard):
        pieces = [torch.empty_like(shard) for _ in range(self.world_size)]
        dist.all_gather(pieces, shard)
        return torch.cat(pieces).view(m.full_shape)

    def _gather_weight(self, m, _):
        m.shard = m.weight.data
        shard = m.shard if self.compute_dtype is None else m.shard.to(self.compute_dtype)
        m.weight.data = self._all_gather(m, shard)

    def _release_weight(self, m, inputs, output):
        m.weight.data = m.shard

    def _make_reduce_scatter_hook(self, m):
        def hook(p):
            grad = p.grad.float().flatten() / self.world_size
            shard_grad = torch.empty_like(m.shard)
            self.handles.append(dist.reduce_scatter_tensor(shard_grad, grad, async_op=True))
            p.data = m.shard
            p.grad = shard_grad

        return hook

    def _all_reduce_grad(self, p):
        p.grad.div_(self.world_size)
        self.handles.append(dist.all_reduce(p.grad, async_op=True))

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()

    def gather_full_params(self):
        params = dict(self.module.named_parameters())
        for name, m in self.module.named_modules():
            if isinstance(m, (Linear, Embedding)):
                params[f"{name}.weight"] = self._all_gather(m, m.weight.data)
        return params
