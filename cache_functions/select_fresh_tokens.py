import torch

def get_cluster_topk_indices(score, cluster_info):
    '''
    找出每个聚类中分数最高的K个索引(用于从每个聚类中找出分数最高的 k 个 token 进行缓存)
    '''
    cluster_indices, cluster_nums, K = cluster_info['cluster_indices'], cluster_info['cluster_nums'], cluster_info['topk']
    cluster_indices, cluster_nums, K = cluster_info['cluster_indices'], cluster_info['cluster_nums'], cluster_info['topk']
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

def get_indices_by_random(cluster_info):
    '''
    选取 K * cluster_nums 个随机索引，相比于从每个聚类中各选 K 个实现更简单，速度更快
    '''
    cluster_indices, cluster_nums, K = cluster_info['cluster_indices'], cluster_info['cluster_nums'], cluster_info['topk']
    B, N = cluster_indices.shape
    device = cluster_indices.device

    fresh_indices = torch.randn((B, N), device=device).argsort(dim=1)[:, :K * cluster_nums]

    return fresh_indices

def select_one_fresh_index_per_cluster(cache_dic, current):
    '''
    select exactly one fresh index for each cluster
    '''
    cluster_info = cache_dic['cluster_info']
    cluster_indices, cluster_nums, K = cluster_info['cluster_indices'], cluster_info['cluster_nums'], cluster_info['topk']
    B, N = cluster_indices.shape
    device = cluster_indices.device
    rand_weights = torch.rand((B, N), device=device)
    cluster_ids = torch.arange(cluster_nums, device=device).view(1, -1, 1)
    mask = (cluster_indices.unsqueeze(1) == cluster_ids)
    masked_weights = torch.where(mask, rand_weights.unsqueeze(1), torch.tensor(-float('inf'), device=device))
    fresh_indices = masked_weights.argmax(dim=2, keepdim=False)
    
    return fresh_indices

def select_fresh_indices_randomly(tokens, topk):
    '''
    随机选择topk个索引(用于和ToCa比较)
    '''
    B, N, D = tokens.shape
    device = tokens.device
    fresh_indices = torch.randn((B, N), device=device).argsort(dim=1)[:, :topk]
    return fresh_indices
