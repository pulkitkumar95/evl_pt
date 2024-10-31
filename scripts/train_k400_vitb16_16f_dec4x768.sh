#!/usr/bin/env sh

exp_dir=runs/k400_vitb16_16f_dec4x768
hostname=$(hostname)
if [[ $hostname == *"arm"* ]]; then
    vid_path=/scratch1/pulkit/kinetics/
    num_nodes=1
    batch_size=64
    num_workers=64
else
    vid_path=/fs/vulcan-datasets/Kinetics-400
    num_nodes=1
    batch_size=8
    num_workers=8
    exp_dir=runs/k400_vitb16_16f_dec4x768_8n
fi


mkdir -p "${exp_dir}"
torchrun --nproc_per_node=${num_nodes} --master_port=19599  \
  main.py \
    --num_steps 50000 \
    --backbone "ViT-B/16-lnpre" \
    --backbone_type clip \
    --backbone_path /fs/cfar-projects/actionloc/bounce_back/efficient-video-recognition/checkpoints/ViT-B-16.pt \
    --decoder_num_layers 4 \
    --decoder_qkv_dim 768 \
    --decoder_num_heads 12 \
    --num_classes 400 \
    --checkpoint_dir "${exp_dir}" \
    --auto_resume \
    --train_list_path /fs/cfar-projects/actionloc/bounce_back/dataset_csvs/k400/train.txt \
    --val_list_path /fs/cfar-projects/actionloc/bounce_back/dataset_csvs/k400/val.txt \
    --batch_size ${batch_size} \
    --batch_split 1 \
    --auto_augment rand-m7-n4-mstd0.5-inc1 \
    --mean 0.48145466 0.4578275 0.40821073 \
    --std 0.26862954 0.26130258 0.27577711 \
    --num_workers ${num_workers} \
    --num_frames 16 \
    --sampling_rate 16 \
    --num_spatial_views 3 \
    --num_temporal_views 1 \
    --vid_base_dir "${vid_path}" --test_batch_size 32 \
    --model_type 'evlbasic' \
    --dataset k400
