import torch
import torch.nn.functional as F

class Kmeans:
    def __init__(self, n_clusters:int, init:str ='random', max_iters=100, p=1):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.init = init
        self.p = p # 1 for Manhattan distance, 2 for Euclid distance

    def fit(self, X, cache_centroids=None):
        if cache_centroids is not None:
            centroids = cache_centroids
        else:
            centroids = self.init_centroids(X)
        for _ in range(self.max_iters):
            dists = torch.cdist(X, centroids, p=self.p)  # [B, N, K]
            labels = dists.argmin(dim=-1)  # [B, N]
            one_hot = F.one_hot(labels, num_classes=self.n_clusters).type_as(X)  # [B, N, K]
            counts = one_hot.sum(dim=1) + 1e-8  # [B, K]
            centroids_new = torch.bmm(
                one_hot.permute(0, 2, 1),  # [B, K, N]
                X                        # [B, N, D]
            ) / counts.unsqueeze(-1)         # [B, K, D]
            if torch.allclose(centroids, centroids_new, rtol=1e-4):
                break
            centroids = centroids_new
        return labels, centroids
    
    def init_centroids(self, X):
        B, N, dim = X.shape
        device = X.device
        if self.init == 'random':
            return X[torch.arange(B, device=device)[:, None], torch.randint(0, N, (B, self.n_clusters), device=device)]
        elif self.init == 'kmeans++':
            # didn't test yet
            return self.kmeans_plusplus(X)
        else:
            raise ValueError(f'Invalid init method: {self.init}')

    def kmeans_plusplus(self, X: torch.Tensor):
        '''
        修正后的 K-means++ 初始化算法
        '''
        B, N, dim = X.shape
        device = X.device
        centers = torch.zeros((B, self.n_clusters, dim), device=device)
        batch_indices = torch.arange(B, device=device)  # (B,)

        # 随机选择第一个中心点
        sample_indices = torch.randint(0, N, (B,), device=device)  # (B,)
        centers[:, 0] = X[batch_indices, sample_indices]

        for i in range(1, self.n_clusters):
            dist = torch.cdist(X, centers[:, :i], p=self.p)  # (B, N, i)
            min_dist, _ = dist.min(dim=-1)  # (B, N)
            min_dist_sq = min_dist ** 2  # 距离平方

            probs = min_dist_sq / (min_dist_sq.sum(dim=1, keepdim=True) + 1e-8)
            probs = probs / probs.sum(dim=1, keepdim=True)

            next_idx = torch.multinomial(probs, 1).squeeze(-1)  # (B,)
            centers[:, i] = X[batch_indices, next_idx]

        return centers