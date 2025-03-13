import torch
def update_cache(fresh_indices, fresh_tokens, cache_dic, current, fresh_attn_map=None):
    '''
    Update the cache with the fresh tokens.
    '''
    step = current['step']
    layer = current['layer']
    module = current['module']
    
    # Update the cached tokens at the positions
    if module == 'attn': 
        # this branch is not used in the final version, but if you explore the partial fresh strategy of attention, it works.
        indices = fresh_indices.sort(dim=1, descending=False)[0]
        
        cache_dic['attn_map'][-1][layer].scatter_(dim=1, index=indices.unsqueeze(-1).expand(-1, -1, fresh_attn_map.shape[-1]), src=fresh_attn_map)
    elif module == 'mlp':
        indices = fresh_indices

    cache_dic['cache'][-1][layer][module].scatter_(dim=1, index=indices.unsqueeze(-1).expand(-1, -1, fresh_tokens.shape[-1]), src=fresh_tokens)
            
def smooth_update_cache(fresh_indices, fresh_tokens, cache_dic, current):
    step = current['step']
    layer = current['layer']
    module = current['module']

    cluster_indices = cache_dic['group_info']['cluster_indices']
    cluster_nums = cache_dic['group_info']['cluster_nums']
    smooth_rate = cache_dic['smooth_rate']
    B, N = cluster_indices.shape
    dim = fresh_tokens.shape[-1]
    tokens = cache_dic['cache'][-1][layer][module]
    device = tokens.device

    fresh_cluster_indices = cluster_indices.gather(dim=1, index=fresh_indices)

    sum_per_cluster = torch.zeros((B, cluster_nums, dim), device=device)
    sum_per_cluster.scatter_add_(
        dim=1,
        index=fresh_cluster_indices.unsqueeze(-1).expand(-1, -1, dim),
        src=fresh_tokens.float()
    )
    
    count_per_cluster = torch.zeros((B, cluster_nums), device=device)
    count_per_cluster.scatter_add_(
        dim=1,
        index=fresh_cluster_indices,
        src=torch.ones_like(fresh_cluster_indices, dtype=torch.float32)
    )

    mean_per_cluster = sum_per_cluster / count_per_cluster.unsqueeze(-1).clamp(min=1e-6)

    expanded_cluster_indices = cluster_indices.unsqueeze(-1).expand(-1, -1, dim)
    new_cache = mean_per_cluster.gather(1, expanded_cluster_indices)

    cand_cache = new_cache * smooth_rate + tokens * (1 - smooth_rate)
    cache_dic['cache'][-1][layer][module] = torch.where(torch.isnan(new_cache), tokens, cand_cache)
    cache_dic['cache'][-1][layer][module].scatter_(dim=1, index=fresh_indices.unsqueeze(-1).expand(-1, -1, dim), src=fresh_tokens)

def smooth_update_cache_v2(fresh_indices, fresh_tokens, cache_dic, current):
    step = current['step']
    layer = current['layer']
    module = current['module']

    group_info = cache_dic['group_info']
    cluster_indices = group_info['cluster_indices']
    cluster_nums = group_info['cluster_nums']
    smooth_rate = cache_dic['smooth_rate']
    B, N = cluster_indices.shape
    dim = fresh_tokens.shape[-1]
    tokens = cache_dic['cache'][-1][layer][module]
    device = tokens.device

    fresh_cluster_indices = cluster_indices.gather(dim=1, index=fresh_indices)
    num_fresh = fresh_indices.shape[1]

    one_hot = torch.zeros((B, num_fresh, cluster_nums), 
                         device=device, 
                         dtype=fresh_tokens.dtype)
    one_hot.scatter_(2, fresh_cluster_indices.unsqueeze(-1), 1.0)
    
    sum_per_cluster = torch.bmm(one_hot.transpose(1, 2), fresh_tokens)
    
    count_per_cluster = one_hot.sum(dim=1)

    mean_per_cluster = sum_per_cluster / count_per_cluster.unsqueeze(-1).clamp(min=1e-6)

    expanded_cluster_indices = cluster_indices.view(B, N, 1).expand(-1, -1, dim)
    new_cache = mean_per_cluster.gather(1, expanded_cluster_indices)

    updated_cache = tokens * (1 - smooth_rate) + new_cache * smooth_rate
    updated_cache = torch.where(torch.isnan(new_cache), tokens, updated_cache)
    
    expanded_indices = fresh_indices.unsqueeze(-1).expand(-1, -1, dim)
    updated_cache.scatter_(1, expanded_indices, fresh_tokens)

    cache_dic['cache'][-1][layer][module] = updated_cache