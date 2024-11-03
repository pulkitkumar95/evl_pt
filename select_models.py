#!/usr/bin/env python

def select_model_func(args):
    model_type = args.model_type
    assert model_type in ['evl', 'oursbasic', 'evlbasic', 'evlselfcrossmotion', 'evltempatt', 'motionrgbfuse']

    print('load models from modeltype {}...'.format(model_type))

    if model_type == 'evl':
        from model import EVLTransformer
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
    
    elif model_type == 'oursbasic':
        from model_basic_ours import OursTransformerBasic 

        model = OursTransformerBasic(
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
        
    elif model_type == 'evlbasic':
        from model_basic_evl import EVLTransformerBasic

        model = EVLTransformerBasic(
        backbone_name=args.backbone,
        backbone_type=args.backbone_type,
        backbone_path=args.backbone_path,
        backbone_mode='finetune' if args.finetune_backbone else ('freeze_fp16' if args.fp16 else 'freeze_fp32'),
        decoder_num_layers=args.decoder_num_layers,
        decoder_qkv_dim=args.decoder_qkv_dim,
        decoder_num_heads=args.decoder_num_heads,
        decoder_mlp_factor=args.decoder_mlp_factor,
        num_classes=args.num_classes,
        cls_dropout=args.cls_dropout,
        decoder_mlp_dropout=args.decoder_mlp_dropout,
        num_frames=args.num_frames,
    )
        
    elif model_type == 'evlselfcrossmotion':
        from model_selfcrossmotion import EVLTransformerSelfCrossmotion

        model = EVLTransformerSelfCrossmotion(
        backbone_name=args.backbone,
        backbone_type=args.backbone_type,
        backbone_path=args.backbone_path,
        backbone_mode='finetune' if args.finetune_backbone else ('freeze_fp16' if args.fp16 else 'freeze_fp32'),
        decoder_num_layers=args.decoder_num_layers,
        decoder_qkv_dim=args.decoder_qkv_dim,
        decoder_num_heads=args.decoder_num_heads,
        decoder_mlp_factor=args.decoder_mlp_factor,
        num_classes=args.num_classes,
        cls_dropout=args.cls_dropout,
        decoder_mlp_dropout=args.decoder_mlp_dropout,
        num_frames=args.num_frames,
    )

    elif model_type == 'evltempatt':
        from model_tempatt import EVLTransformerTempAtt

        model = EVLTransformerTempAtt(
        backbone_name=args.backbone,
        backbone_type=args.backbone_type,
        backbone_path=args.backbone_path,
        backbone_mode='finetune' if args.finetune_backbone else ('freeze_fp16' if args.fp16 else 'freeze_fp32'),
        decoder_num_layers=args.decoder_num_layers,
        decoder_qkv_dim=args.decoder_qkv_dim,
        decoder_num_heads=args.decoder_num_heads,
        decoder_mlp_factor=args.decoder_mlp_factor,
        num_classes=args.num_classes,
        cls_dropout=args.cls_dropout,
        decoder_mlp_dropout=args.decoder_mlp_dropout,
        num_frames=args.num_frames,
    )
        
    elif model_type == 'motionrgbfuse': # 2024.11.03 Sun
        from model_motionrgbfuse import EVLTransformerMotionFuseRGB

        model = EVLTransformerMotionFuseRGB(
        backbone_name=args.backbone,
        backbone_type=args.backbone_type,
        backbone_path=args.backbone_path,
        backbone_mode='finetune' if args.finetune_backbone else ('freeze_fp16' if args.fp16 else 'freeze_fp32'),
        decoder_num_layers=args.decoder_num_layers,
        decoder_qkv_dim=args.decoder_qkv_dim,
        decoder_num_heads=args.decoder_num_heads,
        decoder_mlp_factor=args.decoder_mlp_factor,
        num_classes=args.num_classes,
        cls_dropout=args.cls_dropout,
        decoder_mlp_dropout=args.decoder_mlp_dropout,
        num_frames=args.num_frames,
    )

    
    else:
        raise NotImplementedError()
    
    return model