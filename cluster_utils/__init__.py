import torch
from .Kmeans import Kmeans

def cluster_scheduler(cache_dic, current):
    return cache_dic['cluster_nums'], cache_dic['topk']

def get_cluster_info(X, cache_dic, current):
    cluster_nums, k = cluster_scheduler(cache_dic, current)
    cluster_indices, cache_centroids = Kmeans(n_clusters=cluster_nums, init='random').fit(X, cache_dic['centroids'])
    cache_dic['cluster_info']['cluster_nums'] = cluster_nums
    cache_dic['cluster_info']['topk'] = k
    cache_dic['cluster_info']['cluster_indices'] = cluster_indices
    cache_dic['centroids'] = cache_centroids

def construct_consecutive_cluster_info(X, cache_dic, current):
    '''
    构造连续分组的索引，连续 N//cluster_nums 个token为一组
    '''
    cluster_nums, k = cluster_scheduler(cache_dic, current)
    B, N, D = X.shape
    device = X.device
    segment_length = N // cluster_nums
    cluster_indices = torch.arange(cluster_nums, dtype=torch.long, device=device).repeat_interleave(segment_length)
    cluster_indices = cluster_indices.unsqueeze(0).expand(B, -1)
    cache_dic['cluster_info']['cluster_nums'] = cluster_nums
    cache_dic['cluster_info']['topk'] = k
    cache_dic['cluster_info']['cluster_indices'] = cluster_indices
    
def random_cluster_indices(X, cache_dic, current):
    '''
    随机分组，用于消融聚类
    '''
    cluster_nums, k = cluster_scheduler(cache_dic, current)
    B, N, D = X.shape
    device = X.device
    cluster_indices = torch.randint(0, cluster_nums, (B, N), device=device)
    cache_dic['cluster_info']['cluster_indices'] = cluster_indices
    cache_dic['cluster_info']['cluster_nums'] = cluster_nums
    cache_dic['cluster_info']['topk'] = k
    