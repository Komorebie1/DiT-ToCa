'''
该文件包含了一些用于可视化实验的函数
'''
import PIL
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from scipy.optimize import linear_sum_assignment

def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def visual_batch_cluster(samples, cluster_info, class_labels, dir, vkwargs):
    check_dir(dir)
    cluster_indices = cluster_info['cluster_indices']
    cluster_nums = cluster_info['cluster_nums']
    cols = samples.shape[0]
    samples = samples.sub_(samples.min()).div_(samples.max() - samples.min())
    samples = samples.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
    plt.subplots(2, cols, figsize=(cols * 2, 8))
    plt.subplots_adjust(wspace=0.1, hspace=0.1, left=0.05, right=0.95, bottom=0.05, top=0.95)
    h = w = int(cluster_indices.shape[1] ** 0.5)
    for i in range(cols):
        sample = samples[i]
        cluster_index = cluster_indices[i]
        plt.subplot(2, cols, i + 1)
        plt.imshow(sample)
        plt.axis('off')
        plt.subplot(2, cols, cols + i + 1)
        plt.imshow(cluster_index.reshape(h, w).cpu().numpy())
        plt.axis('off')
    name = "sample"
    for u, v in vkwargs.items():
        name += f"_{u}_{v}"
    plt.savefig(os.path.join(dir, f"{name}.png"))
    plt.close()

def visualize_group(cluster_info, dir):
    check_dir(dir)
    cluster_indices = cluster_info['cluster_indices']
    h = w = int(cluster_indices.shape[1] ** 0.5)
    labels = cluster_indices[0].reshape(h, w).cpu().numpy()
    plt.imshow(labels)
    if not os.path.exists(dir):
        os.makedirs(dir)
    plt.savefig(os.path.join(dir, f'cluster.png'))

def visualize_clustering_difference(cluster_info_list, class_labels, dir):
    check_dir(dir)
    for idx, class_label in enumerate(class_labels):
        difference_list = []
        for i in range(len(cluster_info_list) - 1):
            difference = clustering_difference(cluster_info_list[i], cluster_info_list[i + 1], idx)
            difference_list.append(difference)

        plt.plot(difference_list)
        plt.ylim(0, 1)
        plt.xlabel('Step')
        plt.ylabel('Difference')
        plt.savefig(os.path.join(dir, f'clustering_difference_{class_label}.png'))
        plt.close()
        

def clustering_difference(cluster_info1, cluster_info2, batch_idx):
    labels1 = cluster_info1['cluster_indices'][batch_idx].cpu().numpy()
    labels2 = cluster_info2['cluster_indices'][batch_idx].cpu().numpy()
    k = cluster_info1['cluster_nums']
    
    confusion = np.zeros((k, k), dtype=int)
    for l1, l2 in zip(labels1, labels2):
        confusion[l1, l2] += 1

    row_ind, col_ind = linear_sum_assignment(-confusion)
    max_overlap = confusion[row_ind, col_ind].sum()
    
    total = len(labels1)
    difference = 1.0 - (max_overlap / total)
    return difference

def visualize_score(score, dir, step, layer, topK = 50):
    if isinstance(score, torch.Tensor):
        score = score[0].cpu().numpy()
    plt.figure(figsize=(10, 10))
    topK_indices = np.argsort(score)[::-1][:topK]
    bars = plt.bar(range(len(score)), score, color='blue', alpha=0.7)
    for index in topK_indices:
        bars[index].set_color('red')
    plt.xlabel('Token Index')
    plt.ylabel('Score')
    
    plt.savefig(os.path.join(dir, f'score_step_{step}_layer_{layer}.png'))
    plt.close()

def visualize_cluster_max_min_token_per_dim(tokens:torch.Tensor, cache_dic, current):
    B, N, dim = tokens.shape
    step, layer = current['step'], current['layer']
    cluster_indices, cluster_nums = cache_dic['cluster_info']['cluster_indices'], cache_dic['cluster_info']['cluster_nums']

    for cluster_idx in range(cluster_nums):
        mask = cluster_indices[0] == cluster_idx
        indices = torch.nonzero(mask, as_tuple=True)[0]
        if indices.shape[0] == 0:
            continue
        cluster_i_tokens = tokens[0].gather(0, indices.view(-1, 1).expand(-1, dim))
        # print(cluster_i_tokens.shape)
        cluster_i_tokens_var_per_dim = cluster_i_tokens.var(dim=0)
        if torch.isnan(cluster_i_tokens_var_per_dim).any():
            print(f"Variance computation resulted in NaN for cluster {cluster_idx}.")
            continue
        # print(cluster_i_tokens_var_per_dim.shape)
        data = cluster_i_tokens_var_per_dim.cpu().numpy()
        dir = f"/root/autodl-tmp/exp/token_distribution_var_cluster/cluster_{cluster_idx}/"
        check_dir(dir)
        assert len(data) == dim
        plot_frequency_distribution(data, dir, step, layer, if_log=False)

def visualize_max_min_token_per_dim(tokens, current):
    B, N, dim = tokens.shape
    step, layer = current['step'], current['layer']
    # max_token = tokens.max(dim=1).values
    # min_token = tokens.min(dim=1).values
    # assert max_token.shape == (B, dim) and min_token.shape == (B, dim)

    # value_range = max_token - min_token
    value = tokens.var(dim=1)
    assert value.shape == (B, dim)

    dir = f"/root/autodl-tmp/exp/token_distribution_var/"
    check_dir(dir)
        
    data = value[0].cpu().numpy()
    plot_frequency_distribution(data, dir, step, layer, if_log=False)

def plot_frequency_distribution(data, dir, step, layer, figsize=(10, 6), color='lightblue', density=True, if_log=True):
        """
        绘制数据的频率分布图
        
        参数:
            data (array-like): 输入数据，一维数组或列表
            bins (int): 直方图的分组数量，默认为30
            figsize (tuple): 图形大小，默认为(10, 6)
            color (str): 直方图填充颜色，默认为浅蓝色
            edgecolor (str): 直方图边框颜色，默认为黑色
            density (bool): 是否归一化为频率，默认为True
            show_kde (bool): 是否显示核密度估计曲线，默认为True
        """
        # 创建一个图形和坐标轴
        plt.figure(figsize=figsize)
        ax = plt.gca()
        if if_log:
            data = np.log1p(data)
        # 绘制频率分布图（直方图）
        n, bins, patches = ax.hist(data, bins='auto', density=density, color=color, edgecolor=color, alpha=0.7)
        if if_log:
            ax.set_xlabel('log(varience)', fontsize=12)
        else:
            ax.set_xlabel('varience', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_xlim(left=0, right=10000)
        ax.grid(True, linestyle='--', alpha=0.7)
        # 显示图形
        plt.savefig(os.path.join(dir, f'step_{step}.png'))
        plt.close()

def get_cluster0_entropy(tokens, cache_dic, current, batch_idx=0):
    cluster_indices, cluster_nums = cache_dic['cluster_info']['cluster_indices'], cache_dic['cluster_info']['cluster_nums']
    B, N, dim = tokens.shape
    step, layer = current['step'], current['layer']
    mask = cluster_indices[batch_idx] == 0
    indices = torch.nonzero(mask, as_tuple=True)[0]
    cluster_i_tokens = tokens[batch_idx].gather(0, indices.view(-1, 1).expand(-1, dim))
    entropy = entropy_kde(cluster_i_tokens)
    return entropy

def get_tokens_entropy(tokens):
    B, N, dim = tokens.shape
    tokens = tokens[0]
    entropy = entropy_kde(tokens)
    return entropy
        

def entropy_kde(x: torch.Tensor, h: float = None, eps: float = 1e-10) -> torch.Tensor:
    """
    用PyTorch计算[N, dim]向量的信息熵（基于核密度估计KDE）
    
    Args:
        x: 输入张量，形状为 [N, dim]
        h: 高斯核带宽（默认为Scott规则自动计算）
        eps: 数值稳定项，避免log(0)
    
    Returns:
        信息熵值（标量）
    """
    N, dim = x.shape
    
    # 自动计算带宽（Scott规则）
    if h is None:
        std_per_dim = torch.std(x, dim=0, unbiased=False)  # 每个维度的标准差
        mean_std = std_per_dim.mean()  # 平均标准差
        scott_factor = (N ** (-1.0 / (dim + 4)))  # Scott带宽因子
        h = mean_std * scott_factor
    
    # 计算所有样本对的平方欧氏距离
    x_norm = (x ** 2).sum(dim=1)  # [N]
    dists = x_norm.unsqueeze(1) + x_norm.unsqueeze(0) - 2 * torch.mm(x, x.t())
    dists = torch.clamp(dists, min=0.0)  # 确保距离非负
    
    # 高斯核密度估计
    kernel_vals = torch.exp(-dists / (2 * h**2))  # [N, N]
    densities = kernel_vals.sum(dim=1) / (N * (h ** dim) + eps)  # [N]
    
    # 计算熵: H = -E[log(p(x))]
    entropy = -torch.log(densities + eps).mean()
    return entropy

def get_entropy(tokens, cache_dic, current, batch_idx=0):
    if cache_dic['cluster_method'] == 'DBSCAN':
        return get_entropy_for_DBSCAN(tokens, cache_dic, current)
    cluster_indices, cluster_nums = cache_dic['cluster_info']['cluster_indices'], cache_dic['cluster_info']['cluster_nums']
    B, N, dim = tokens.shape
    step, layer = current['step'], current['layer']
    cluster_indices0 = cluster_indices[0]
    entropy = 0
    for cluster_idx in range(cluster_nums):
        mask = cluster_indices0==cluster_idx
        count = mask.sum()
        if count == 0:
            continue
        entropy -= count/N * torch.log2(count/N)
    return entropy

def get_entropy_for_DBSCAN(x, cache_dic, current):
    cluster_indices = cache_dic['cluster_info']['cluster_indices']
    cluster_indices = cluster_indices[0].cpu().numpy()
    _, counts = np.unique(cluster_indices, return_counts=True)
    total_tokens = len(cluster_indices)
    entropy = 0
    
    for count in counts:
        if count > 0:  # 排除噪声点
            p_k = count / total_tokens
            entropy -= p_k * np.log2(p_k)
    return entropy