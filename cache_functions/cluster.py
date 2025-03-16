import torch
import torch.nn.functional as F
import numpy as np

def cluster_scheduler(cache_dic, current):
    if not cache_dic['use_cluster_scheduler']:
        return cache_dic['cluster_nums'], cache_dic['topk']
    return int(cache_dic['current_cluster_nums'][current['step']]), round(5 - current['step'] / 12)

def get_cluster_info(features, cache_dic, current):
    cluster_nums, k = cluster_scheduler(cache_dic, current)
    cache_dic['cluster_info']['cluster_nums'] = cluster_nums
    cache_dic['cluster_info']['topk'] = k
    cache_dic['cluster_info']['cluster_indices'] = torch.zeros((features.shape[0], features.shape[1]), device=features.device, dtype=torch.int64)
    if k == 0:
        return
    cluster_indices, cache_centroids = get_cluster_indices_by_features(features, cluster_nums, cache_dic['cluster_method'], None)
    cache_dic['cluster_info']['cluster_indices'] = cluster_indices
    cache_dic['centroids'] = cache_centroids


def get_cluster_indices_by_features(features, cluster_nums, cluster_method, cache_centroids=None):
    '''
    计算聚类索引
    '''
    coords = features
    if cluster_method in ['kmeans', 'kmeans++']:
        if cache_centroids is None and cluster_method == 'kmeans++':
            cache_centroids = kmeans_plusplus(coords, cluster_nums)
        return kmeans_clustering(coords, cluster_nums, cache_centroids=cache_centroids, p=1)
    elif cluster_method == 'random':
        return random_cluster_indices(features.shape[0], features.shape[1], cluster_nums, device=features.device), None
    else:
        raise ValueError(f'Invalid cluster method: {cluster_method}')

def construct_cluster_indices_with_padding(B, N, C, device):
    '''
    构造连续分组的索引，连续 N//C 个token为一组
    '''
    segment_length = N // C
    cluster_indices = torch.arange(C, dtype=torch.long, device=device).repeat_interleave(segment_length)
    cluster_indices = cluster_indices.unsqueeze(0).expand(B, -1)

    return cluster_indices

def random_cluster_indices(B, N, C, device):
    '''
    随机分组，用于消融聚类
    '''
    cluster_indices = torch.randint(0, C, (B, N), device=device)
    return cluster_indices

def kmeans_clustering(features, cluster_num, cache_centroids=None, max_iters=100, p=1):
    B, N, D = features.shape
    device = features.device
    if cache_centroids is not None:
        centroids = cache_centroids
    else:
        centroids = features[torch.arange(B, device=device)[:, None], torch.randint(0, N, (B, cluster_num), device=device)]
    
    for _ in range(max_iters):
        dists = torch.cdist(features, centroids, p=p)  # [B, N, K]
        labels = dists.argmin(dim=-1)  # [B, N]
        one_hot = F.one_hot(labels, num_classes=cluster_num).type_as(features)  # [B, N, K]
        counts = one_hot.sum(dim=1) + 1e-8  # [B, K]
        centroids_new = torch.bmm(
            one_hot.permute(0, 2, 1),  # [B, K, N]
            features                        # [B, N, D]
        ) / counts.unsqueeze(-1)         # [B, K, D]
        if torch.allclose(centroids, centroids_new, rtol=1e-4):
            break
        centroids = centroids_new
    
    return labels, centroids

def kmeans_plusplus(features: torch.Tensor, cluster_nums: int):
    '''
    修正后的 K-means++ 初始化算法
    '''
    B, N, dim = features.shape
    device = features.device
    centers = torch.zeros((B, cluster_nums, dim), device=device)
    batch_indices = torch.arange(B, device=device)  # (B,)

    # 随机选择第一个中心点
    sample_indices = torch.randint(0, N, (B,), device=device)  # (B,)
    centers[:, 0] = features[batch_indices, sample_indices]

    for i in range(1, cluster_nums):
        # 计算每个点到最近中心的距离（平方）
        dist = torch.cdist(features, centers[:, :i], p=1)  # (B, N, i)
        min_dist, _ = dist.min(dim=-1)  # (B, N)
        min_dist_sq = min_dist ** 2  # 距离平方

        probs = min_dist_sq / (min_dist_sq.sum(dim=1, keepdim=True) + 1e-8)
        probs = probs / probs.sum(dim=1, keepdim=True)

        next_idx = torch.multinomial(probs, 1).squeeze(-1)  # (B,)
        centers[:, i] = features[batch_indices, next_idx]

    return centers