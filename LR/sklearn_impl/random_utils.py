"""手写 sklearn 随机规则；避免文件名遮蔽 Python 标准库 _random。"""

UINT32_MASK = (1 << 32) - 1
RAND_R_MAX = (1 << 31) - 1


def our_rand_r(state):
    state = int(state) or 1
    state ^= (state << 13) & UINT32_MASK
    state ^= state >> 17
    state ^= (state << 5) & UINT32_MASK
    state &= UINT32_MASK
    return state, state % (RAND_R_MAX + 1)


def shuffle_indices(indices, seed):
    """每次调用重置局部 seed，但继续置换已有 indices。"""
    state = seed
    size = len(indices)
    for i in range(size - 1):
        state, random_value = our_rand_r(state)
        j = i + random_value % (size - i)
        indices[i], indices[j] = indices[j], indices[i]
