import torch

class KVCache:
    """
    一个简单且高效的 Key-Value 缓存类。

    这个类为自回归模型的每个注意力层存储 key 和 value 的状态。
    它被设计为在每次生成新 token 时进行高效的更新。
    """
    def __init__(self):
        self.key_cache = None
        self.value_cache = None

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor):
        """
        用新的 key 和 value 状态更新缓存。

        Args:
            key_states (torch.Tensor): 当前步骤计算出的 key 张量。
            value_states (torch.Tensor): 当前步骤计算出的 value 张量。
        """
        if self.key_cache is None or self.value_cache is None:
            self.key_cache = key_states.detach()
            self.value_cache = value_states.detach()
        else:
            self.key_cache = torch.cat([self.key_cache, key_states.detach()], dim=2)
            self.value_cache = torch.cat([self.value_cache, value_states.detach()], dim=2)

    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        获取当前完整的 key 和 value 缓存。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (完整的 key 缓存, 完整的 value 缓存)
        """
        return self.key_cache, self.value_cache

    def clear(self):
        """
        清空缓存，在开始一次新的、独立的生成任务时调用。
        """
        self.key_cache = None
        self.value_cache = None

    def __len__(self) -> int:
        """
        返回缓存的当前序列长度。
        """
        if self.key_cache is None:
            return 0
        # key_cache 的形状是 [batch, n_heads, seq_len, head_dim]
        return self.key_cache.shape[2]