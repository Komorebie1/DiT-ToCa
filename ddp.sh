#!/bin/bash

model="DiT-XL/2"
per_proc_batch_size=100
image_size=256
cfg_scale=1.5
num_sampling_steps=50
cache_type="attention"
fresh_ratio=0.07
ratio_scheduler="ToCa-ddim50"
force_fresh="global"
fresh_threshold=4
soft_fresh_weight=0.25
num_fid_samples=50000
cluster_nums=16
cluster_method="kmeans"

model_string_name="DiT-XL-2"
ckpt_string_name="pretrained"
args_vae="mse"
args_global_seed=0
topk=1

base_command="torchrun \
    --nnodes=1 \
    --nproc_per_node=6 \
    sample_ddp.py \
    --model $model \
    --per-proc-batch-size $per_proc_batch_size \
    --image-size $image_size \
    --cfg-scale $cfg_scale \
    --num-sampling-steps $num_sampling_steps \
    --cache-type $cache_type \
    --fresh-ratio $fresh_ratio \
    --ratio-scheduler $ratio_scheduler \
    --force-fresh $force_fresh \
    --fresh-threshold $fresh_threshold \
    --soft-fresh-weight $soft_fresh_weight \
    --num-fid-samples $num_fid_samples \
    --ddim-sample \
    --cluster-nums $cluster_nums \
    --cluster-method $cluster_method \
    --topk $topk \
    "

# cluster_nums=(4 8 16 16 16 16 16 16 16 16 32 32)
# topks=(20 10 5 2 1 10 1 1 1 1 1 1)
# momentum_rates=(0.0 0.0 0.007 0.007 0.007 0.007 0.03 0.02 0.006 0.008 0.007 0.005)
# momentum_rates=(0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
smooth_rate=0.007

for ((i=0;i<1;i++))
do
    # rate=${momentum_rates[i]}
    rate=0.007
    echo "running with smooth_rates: $rate"
    eval $base_command --smooth-rate $rate

    if [ $? -eq 0 ]; then
        /root/miniconda3/envs/eval/bin/python evaluator.py /root/autodl-tmp/VIRTUAL_imagenet256_labeled.npz \
            "/root/autodl-tmp/samples/ToCa-${cluster_nums}-${cluster_method}-${topk}-${rate}.npz"
    else
        echo "torchrun 命令执行失败，跳过 evaluator.py"
    fi

    if [ $? -eq 0 ]; then
        rm -rf "/root/autodl-tmp/samples/ToCa-${cluster_nums}-${cluster_method}-${topk}-${rate}"
        rm  "/root/autodl-tmp/samples/ToCa-${cluster_nums}-${cluster_method}-${topk}-${rate}.npz"
    else
        echo "evaluator.py 执行失败，跳过删除文件"
    fi
done