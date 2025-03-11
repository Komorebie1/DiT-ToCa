import torch

def get_cluster_topk_indices(score, group_info):
    '''
    找出每个聚类中分数最高的K个索引(用于从每个聚类中找出分数最高的 k 个 token 进行缓存)
    '''
    cluster_indices, cluster_nums, K = group_info['cluster_indices'], group_info['cluster_nums'], group_info['topK']
    B, N = score.shape
    device = score.device
    
    k_indices = torch.arange(cluster_nums, device=device).view(1, cluster_nums, 1)
    mask = (cluster_indices.unsqueeze(1) == k_indices)

    score_masked = score.unsqueeze(1).masked_fill(~mask, -float('inf'))
    _, topk_indices = torch.topk(score_masked, K, dim=-1)
    cluster_count = mask.sum(dim=-1)

    # 处理空聚类和不足K的情况
    valid_mask = (torch.arange(K, device=device).view(1, 1, -1) < cluster_count.unsqueeze(-1))
    topk_indices = torch.where(
        valid_mask | (cluster_count == 0).unsqueeze(-1),
        topk_indices,
        torch.zeros_like(topk_indices)
    )
    
    return topk_indices.view(B, -1)

def get_indices_by_random(group_info):
    cluster_indices, cluster_nums, K = group_info['cluster_indices'], group_info['cluster_nums'], group_info['topK']
    B, N = cluster_indices.shape
    device = cluster_indices.device

    fresh_indices = torch.randn((B, N), device=device).argsort(dim=1)[:, :K * cluster_nums]

    return fresh_indices

def get_indices_by_random_v2(group_info):
    cluster_indices, cluster_nums, K = group_info['cluster_indices'], group_info['cluster_nums'], group_info['topK']
    B, N = cluster_indices.shape
    device = cluster_indices.device    
    # 生成聚类掩码 [B, cluster_nums, N]
    cluster_mask = (cluster_indices.unsqueeze(1) == torch.arange(cluster_nums, device=device).view(1, -1, 1))
    # 阶段1：优先选择聚类内元素
    # 生成随机排序索引
    rand = torch.rand((B, cluster_nums, N), device=device)
    masked_rand = torch.where(cluster_mask, rand, torch.tensor(float('inf'), device=device))
    sorted_indices = torch.argsort(masked_rand, dim=2)  # [B, cluster_nums, N]
    # 阶段2：处理元素不足的情况
    # 计算每个聚类的实际元素数量 [B, cluster_nums]
    counts = cluster_mask.sum(dim=2)  # [B, cluster_nums]
    # 生成基础索引模板
    base_idx = torch.arange(K, device=device).view(1, 1, -1).expand(B, cluster_nums, -1)
    # 计算循环索引
    mod_counts = torch.where(counts > 0, counts, torch.ones_like(counts))
    cyclic_idx = base_idx % mod_counts.unsqueeze(-1)  # [B, cluster_nums, k]
    # 收集聚类内选择结果 [B, cluster_nums, k]
    intra_selected = torch.gather(sorted_indices, 2, cyclic_idx)
    # 阶段3：处理空聚类和补充采样
    # 生成全局随机索引 [B, cluster_nums, k]
    global_random = torch.randint(0, N, (B, cluster_nums, K), device=device)
    # 创建选择掩码 [B, cluster_nums, k]
    valid_mask = (counts.unsqueeze(-1) > base_idx)  # 有效位置掩码
    # 合并结果
    final_selected = torch.where(valid_mask, intra_selected, global_random)
    # 展平为[B, cluster_nums*k]
    return final_selected.view(B, -1)