import torch
import torch.nn.functional as F
import numpy as np

def cluster_scheduler(cache_dic, current):
    if not cache_dic['use_cluster_scheduler']:
        return cache_dic['cluster_nums'], cache_dic['topK']
    return int(cache_dic['current_cluster_nums'][current['step']]), round(5 - current['step'] / 12)

def get_group_info(cache_dic, current, dims=2):
    cluster_nums, k = cluster_scheduler(cache_dic, current)
    # cluster_indices = get_group_indices(cache_dic['key_matrix'], cluster_nums, dims=dims, cluster_method=cache_dic['cluster_method'])
    cluster_indices, cache_centroids = get_group_indices_by_x(cache_dic['x'], cluster_nums, cache_dic['cluster_method'], cache_dic['centroids'])
    cache_dic['group_info']['cluster_indices'] = cluster_indices
    cache_dic['group_info']['cluster_nums'] = cluster_nums
    cache_dic['group_info']['topK'] = k
    cache_dic['centroids'] = cache_centroids


def get_group_indices_by_x(token, cluster_nums, cluster_method, cache_centroids=None):
    '''
    计算聚类索引
    '''
    # coords = tsne(token, dims)
    coords = token
    if cluster_method == 'kmeans':
        cluster_indices, centroids = kmeans_clustering(coords, cluster_nums, cache_centroids=cache_centroids, p=1)
    elif cluster_method in ['spectral', 'Agglomerative', 'DBSCAN']:
        cluster_indices = cluster_by_sklearn_for_token(coords, cluster_nums, cluster_method)
    else:
        raise ValueError(f'Invalid cluster method: {cluster_method}')
    return cluster_indices, centroids


def kmeans_clustering(tokens, cluster_num, cache_centroids=None, max_iters=100, p=2):
    B, N, D = tokens.shape
    device = tokens.device
    if cache_centroids is not None:
        centroids = cache_centroids
    else:
        centroids = tokens[torch.arange(B, device=device)[:, None], torch.randint(0, N, (B, cluster_num), device=device)]
    
    for _ in range(max_iters):
        dists = torch.cdist(tokens, centroids, p=p)  # [B, N, K]
        labels = dists.argmin(dim=-1)  # [B, N]
        one_hot = F.one_hot(labels, num_classes=cluster_num).type_as(tokens)  # [B, N, K]
        counts = one_hot.sum(dim=1) + 1e-8  # [B, K]
        centroids_new = torch.bmm(
            one_hot.permute(0, 2, 1),  # [B, K, N]
            tokens                        # [B, N, D]
        ) / counts.unsqueeze(-1)         # [B, K, D]
        if torch.allclose(centroids, centroids_new, rtol=1e-4):
            break
        centroids = centroids_new
    
    return labels, centroids

def cluster_by_sklearn(similarity_matrix, cluster_nums, method = "Agglomerative"):
    import sklearn.cluster
    B, N, _ = similarity_matrix.shape
    device = similarity_matrix.device
    similarity_matrix = (1 - similarity_matrix).cpu().numpy()

    cluster_labels = np.zeros((B, N), dtype=np.int32)
    if method == "spectral":
        cluster = sklearn.cluster.SpectralClustering(n_clusters=cluster_nums, affinity='precomputed')
        for b_idx in range(B):
            labels = cluster.fit(similarity_matrix[b_idx]).labels_
            cluster_labels[b_idx] = labels
    elif method == "Agglomerative":
        cluster = sklearn.cluster.AgglomerativeClustering(n_clusters=cluster_nums, metric='precomputed', linkage='average', distance_threshold=None)
        for b_idx in range(B):
            labels = cluster.fit(similarity_matrix[b_idx]).labels_
            cluster_labels[b_idx] = labels
    elif method == "DBSCAN":
        cluster = sklearn.cluster.DBSCAN(eps=0.5, min_samples=5, metric='precomputed')
        for b_idx in range(B):
            labels = cluster.fit(similarity_matrix[b_idx]).labels_
            cluster_labels[b_idx] = labels
    else:
        raise ValueError(f'Invalid cluster method: {method}')

    return torch.tensor(cluster_labels, device=device)

def cluster_by_sklearn_for_token(token, cluster_nums, method = "Agglomerative"):
    import sklearn.cluster
    B, N, _ = token.shape
    device = token.device
    token = token.cpu().numpy()
    cluster_labels = np.zeros((B, N), dtype=np.int32)
    if method == "spectral":
        cluster = sklearn.cluster.SpectralClustering(n_clusters=cluster_nums)
        for b_idx in range(B):
            labels = cluster.fit(token[b_idx]).labels_
            cluster_labels[b_idx] = labels
    elif method == "Agglomerative":
        cluster = sklearn.cluster.AgglomerativeClustering(n_clusters=cluster_nums, linkage='average', distance_threshold=None, metric='l1')
        for b_idx in range(B):
            labels = cluster.fit(token[b_idx]).labels_
            cluster_labels[b_idx] = labels
        # from cuml.cluster import AgglomerativeClustering
        # agg = AgglomerativeClustering(n_clusters=cluster_nums, metric = 'l1', linkage='average')
        # cluster_labels = agg.fit_predict(token)
    elif method == "DBSCAN":
        cluster = sklearn.cluster.DBSCAN(eps=0.5, min_samples=5)
        for b_idx in range(B):
            labels = cluster.fit(token[b_idx]).labels_
            cluster_labels[b_idx] = labels
    else:
        raise ValueError(f'Invalid cluster method: {method}')

    return torch.tensor(cluster_labels, device=device)

def batch_multi_dim_scale(similarity_matrix, num_dims=2, eps=1e-12):
    '''
    使用截断SVD实现多维度缩放
    '''
    # 优化后的矩阵运算流程
    B, N, _ = similarity_matrix.shape
    device = similarity_matrix.device
    
    # 使用更高效的距离计算
    distance_matrix = 2 * (1 - similarity_matrix.sigmoid()).sqrt_().add_(eps)
    
    # 中心化矩阵（利用广播机制）
    H = torch.eye(N, device=device) - 1/N
    B_matrix = -0.5 * H @ (distance_matrix**2) @ H
    
    # 使用截断SVD加速计算
    U, S, _ = torch.svd_lowrank(B_matrix, q=num_dims+2, niter=3)
    coords = U[:, :, :num_dims] * torch.sqrt(S[:, :num_dims]).unsqueeze(1)
    # print(coords.shape)
    return coords

def tsne(token, dims):
    from sklearn.manifold import TSNE
    B, N, _ = token.shape
    device = token.device
    coords = np.zeros((B, N, dims))
    for b in range(B):
        X_tsne = TSNE(n_components=dims, perplexity=5).fit_transform(token[b].cpu().numpy())
        coords[b] = X_tsne
    return torch.tensor(coords, device=device)