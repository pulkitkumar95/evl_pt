#!/usr/bin/env python
import os
import argparse
import datetime as dt
from datetime import datetime
import builtins
import wandb
import torch
import torch.distributed as dist

import video_dataset
import checkpoint
from model import EVLTransformer
from video_dataset import dataloader
from weight_loaders import weight_loader_fn_dict
from vision_transformer import vit_presets

OG_BATCH_SIZE = 256




def new_dist_init(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True
    args.init_method = 'env://'

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(
        "| distributed init (rank {}, world {}): {}".format(
            args.rank, args.world_size, args.init_method
        ),
        flush=True,
    )
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.init_method,
        world_size=args.world_size,
        rank=args.rank,
        timeout=dt.timedelta(
            days=365
        ),  # allow auto-downloading and de-compressing
    )
    print('Done init')
    torch.distributed.barrier()
    return args

def setup_wandb(args: argparse.Namespace):
    os.environ["WANDB_RESUME"] = "allow"
    os.environ["WANDB_RUN_ID"] = args.wandb_id if args.wandb_id is not None else wandb.util.generate_id()
    wandb_run = wandb.init(project='evl_pt',config=args.__dict__, 
                                    entity="act_seg_pi_umd")
    wandb_run.define_metric("epoch")
    wandb_run.define_metric("iteration")

    wandb_run.define_metric("iter*", step_metric="iteration")

    wandb_run.define_metric("train*", step_metric="epoch")
    wandb_run.define_metric("lr", step_metric="epoch")

    wandb_run.define_metric("val*", step_metric="epoch")
    wandb_run.define_metric("train_loss", summary="min")
    wandb_run.define_metric("val_loss", summary="min")
    wandb_run.define_metric("val_top5_acc", summary="max")
    wandb_run.define_metric("val_top1_acc", summary="max")
    return wandb_run

def setup_print(is_master: bool):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            now = datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def main():
    parser = argparse.ArgumentParser()
    
    video_dataset.setup_arg_parser(parser)
    checkpoint.setup_arg_parser(parser)

    parser.add_argument('--num_steps', type=int,
                        help='number of training steps')
    parser.add_argument('--eval_only', action='store_true',
                        help='run evaluation only')
    parser.add_argument('--save_freq', type=int, default=5000,
                        help='save a checkpoint every N steps')
    parser.add_argument('--eval_freq', type=int, default=5000,
                        help='evaluate every N steps')
    parser.add_argument('--print_freq', type=int, default=10,
                        help='print log message every N steps')

    parser.add_argument('--backbone', type=str, choices=vit_presets.keys(), default='ViT-B/16-lnpre',
                        help='the backbone variant used to generate image feature maps')
    parser.add_argument('--backbone_path', type=str,
                        help='path to pretrained backbone weights')
    parser.add_argument('--backbone_type', type=str, default='clip', choices=weight_loader_fn_dict.keys(),
                        help='type of backbone weights (used to determine how to convert state_dict from different pretraining codebase)')
    parser.add_argument('--finetune_backbone', action='store_true',
                        help='finetune backbone weights')
    parser.add_argument('--decoder_num_layers', type=int, default=4,
                        help='number of decoder layers')
    parser.add_argument('--decoder_qkv_dim', type=int, default=768,
                        help='q (k, v) projection output dimensions in decoder attention layers')
    parser.add_argument('--decoder_num_heads', type=int, default=12,
                        help='number of heads in decoder attention layers')
    parser.add_argument('--decoder_mlp_factor', type=float, default=4.0,
                        help='expansion factor of feature dimension in the middle of decoder MLPs')
    parser.add_argument('--num_classes', type=int, default=400,
                        help='number of classes')
    parser.add_argument('--cls_dropout', type=float, default=0.5,
                        help='dropout rate applied before the final classification linear projection')
    parser.add_argument('--decoder_mlp_dropout', type=float, default=0.5,
                        help='dropout rate applied in MLP layers in the decoder')
    parser.add_argument('--no_temporal_conv', action='store_false', dest='temporal_conv',
                        help='disable temporal convolution on frame features')
    parser.add_argument('--no_temporal_pos_embed', action='store_false', dest='temporal_pos_embed',
                        help='disable temporal position embeddings added to frame features')
    parser.add_argument('--no_temporal_cross_attention', action='store_false', dest='temporal_cross_attention',
                        help='disable temporal cross attention on frame query and key features')
    parser.set_defaults(temporal_conv=True, temporal_pos_embed=True, temporal_cross_attention=True)

    parser.add_argument('--lr', type=float, default=4e-4,
                        help='learning rate')
    parser.add_argument('--test_batch_size', type=int, default=1,
                        help='batch size for validation')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='optimizer weight decay')
    parser.add_argument('--disable_fp16', action='store_false', dest='fp16',
                        help='disable fp16 during training or inference')
    parser.set_defaults(fp16=True)

    parser.add_argument('--batch_split', type=int, default=1,
                        help='optionally split the batch into smaller shards and forward/backward one shard '
                             'at a time to avoid out-of-memory error.')
    parser.add_argument('--vid_base_dir', type=str, default='/fs/vulcan-datasets/Kinetics-400/',
                        help='base directory for video files')
    parser.add_argument('--wandb_id', type=str, default=None,
                        help='wandb id for logging')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='experiment name for logging')
    
    parser.add_argument('--dataset', type=str, required=True,
                        help='dataset name')
    args = parser.parse_args()

    args = new_dist_init(args)
    setup_print(dist.get_rank() == 0)
    
    # adusting lr and iterations basded on batch size, this is to keep the total training time the same
    batch_size_ratio = args.batch_size / OG_BATCH_SIZE
    args.lr *= batch_size_ratio**0.5
    args.num_steps = int(args.num_steps / batch_size_ratio)
    args.eval_freq = int(args.eval_freq / batch_size_ratio)
    args.save_freq = int(args.save_freq / batch_size_ratio)



    cuda_device_id = args.gpu
    print(f"Using GPU {cuda_device_id}")
    torch.cuda.set_device(cuda_device_id)
    if args.gpu == 0:
        wandb_run = setup_wandb(args)
    else:
        wandb_run = None
    

    model = EVLTransformer(
        backbone_name=args.backbone,
        backbone_type=args.backbone_type,
        backbone_path=args.backbone_path,
        backbone_mode='finetune' if args.finetune_backbone else ('freeze_fp16' if args.fp16 else 'freeze_fp32'),
        decoder_num_layers=args.decoder_num_layers,
        decoder_qkv_dim=args.decoder_qkv_dim,
        decoder_num_heads=args.decoder_num_heads,
        decoder_mlp_factor=args.decoder_mlp_factor,
        num_classes=args.num_classes,
        enable_temporal_conv=args.temporal_conv,
        enable_temporal_pos_embed=args.temporal_pos_embed,
        enable_temporal_cross_attention=args.temporal_cross_attention,
        cls_dropout=args.cls_dropout,
        decoder_mlp_dropout=args.decoder_mlp_dropout,
        num_frames=args.num_frames,
    )
    model.cuda(cuda_device_id)

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[cuda_device_id], output_device=cuda_device_id,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_steps)
    loss_scaler = torch.cuda.amp.grad_scaler.GradScaler(enabled=args.fp16)
    criterion = torch.nn.CrossEntropyLoss()

    resume_step = checkpoint.resume_from_checkpoint(model, optimizer, lr_sched, loss_scaler, args)

    val_loader = video_dataset.create_val_loader(args)
    if args.eval_only:
        print('Running in eval_only mode.')
        model.eval()
        evaluate(model, val_loader)
        return
    else:
        assert args.train_list_path is not None, 'Train list path must be specified if not in eval_only mode.'
        train_loader = video_dataset.create_train_loader(args, resume_step=resume_step)

    assert len(train_loader) == args.num_steps - resume_step
    batch_st, train_st = datetime.now(), datetime.now()
    for i, (data, labels) in enumerate(train_loader, resume_step):
        data, labels = data.cuda(), labels.cuda()
        data_ed = datetime.now()

        optimizer.zero_grad()

        assert data.size(0) % args.batch_split == 0
        split_size = data.size(0) // args.batch_split
        hit1, hit5, loss_value = 0, 0, 0
        for j in range(args.batch_split):
            data_slice = data[split_size * j: split_size * (j + 1)]
            labels_slice = labels[split_size * j: split_size * (j + 1)]

            with torch.cuda.amp.autocast(args.fp16):
                logits = model(data_slice)
                loss = criterion(logits, labels_slice)
                
            if labels.dtype == torch.long: # no mixup, can calculate accuracy
                hit1 += (logits.topk(1, dim=1)[1] == labels_slice.view(-1, 1)).sum().item()
                hit5 += (logits.topk(5, dim=1)[1] == labels_slice.view(-1, 1)).sum().item()
            loss_value += loss.item() / args.batch_split
            
            loss_scaler.scale(loss / args.batch_split).backward()
        
        loss_scaler.step(optimizer)
        loss_scaler.update()
        lr_sched.step()

        batch_ed = datetime.now()

        if i % args.print_freq == 0:
            sync_tensor = torch.Tensor([loss_value, hit1 / data.size(0), hit5 / data.size(0)]).cuda()
            dist.all_reduce(sync_tensor)
            sync_tensor = sync_tensor.cpu() / dist.get_world_size()
            loss_value, acc1, acc5 = sync_tensor.tolist()
            if wandb_run is not None:
                wandb_run.log({
                    "train_loss": loss_value,
                    "train_acc1": acc1,
                    "train_acc5": acc5,
                })

            print(
                f'batch_time: {(batch_ed - batch_st).total_seconds():.3f}  '
                f'data_time: {(data_ed - batch_st).total_seconds():.3f}  '
                f'ETA: {(batch_ed - train_st) / (i - resume_step + 1) * (args.num_steps - i - 1)}  |  '
                f'lr: {optimizer.param_groups[0]["lr"]:.6f}  '
                f'loss: {loss_value:.6f}' + (
                    f'  acc1: {acc1 * 100:.2f}%  acc5: {acc5 * 100:.2f}%' if labels.dtype == torch.long else ''
                )
            )
        if (i + 1) % args.save_freq == 0:
            checkpoint.save_checkpoint(model, optimizer, lr_sched, loss_scaler, i + 1, args)    
        if (i + 1) % args.eval_freq == 0:
            print('Start model evaluation at step', i + 1)
            model.eval()
            evaluate(model, val_loader, wandb_run)
            model.train()

        
        
        batch_st = datetime.now()


def evaluate(model: torch.nn.Module, loader: torch.utils.data.DataLoader, wandb_run=None):
    tot, hit1, hit5 = 0, 0, 0
    eval_st = datetime.now()
    for batch_idx, (data, labels) in enumerate(loader):
        data, labels = data.cuda(), labels.cuda()
        batch_size, num_views = data.shape[:2]
        data = data.view(batch_size * num_views, *data.shape[2:])
        labels = labels.unsqueeze(-1)

        with torch.no_grad():
            logits = model(data)
            scores = logits.softmax(dim=-1)
            scores = scores.view(batch_size, num_views, -1)
            scores = scores.mean(dim=1)

        tot += batch_size
        hit1 += (scores.topk(1)[1] == labels).sum().item()
        hit5 += (scores.topk(5)[1] == labels).sum().item()

        if tot % 20 == 0:
            print(f'[Evaluation] num_samples: {tot}  '
                  f'ETA: {(datetime.now() - eval_st) / batch_idx * (len(loader) - batch_idx)}  '
                  f'cumulative_acc1: {hit1 / tot * 100.:.2f}%  '
                  f'cumulative_acc5: {hit5 / tot * 100.:.2f}%')

    sync_tensor = torch.LongTensor([tot, hit1, hit5]).cuda()
    dist.all_reduce(sync_tensor)
    tot, hit1, hit5 = sync_tensor.cpu().tolist()
    if wandb_run is not None:
        wandb_run.log({
            "val_top1_acc": hit1 / tot,
            "val_top5_acc": hit5 / tot,
        })

    print(f'Accuracy on validation set: top1={hit1 / tot * 100:.2f}%, top5={hit5 / tot * 100:.2f}%')


if __name__ == '__main__': main()
