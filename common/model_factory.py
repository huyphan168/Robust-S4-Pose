import os
from common.model_conf_scr import *
def get_model(args, num_joints_in, num_joints_out, in_features):
    filter_widths = [int(x) for x in args.architecture.split(',')]
    if args.model == "VideoPose3D":
        from common.model import TemporalModel, TemporalModelOptimized1f
        # Prepare model
        if not args.disable_optimizations and not args.dense and args.stride == 1:
            # Use optimized model for single-frame predictions
            model_pos_train = TemporalModelOptimized1f(num_joints_in, in_features, num_joints_out,
                                        filter_widths=filter_widths, causal=args.causal, dropout=args.dropout, channels=args.channels)
        else:
            # When incompatible settings are detected (stride > 1, dense filters, or disabled optimization) fall back to normal model
            model_pos_train = TemporalModel(num_joints_in, in_features, num_joints_out,
                                        filter_widths=filter_widths, causal=args.causal, dropout=args.dropout, channels=args.channels,
                                        dense=args.dense)
            
        model_pos = TemporalModel(num_joints_in, in_features, num_joints_out,
                                    filter_widths=filter_widths, causal=args.causal, dropout=args.dropout, channels=args.channels,
                                    dense=args.dense)
        return model_pos_train, model_pos
    
    elif args.model == 'SRNet':
        from common.model_srnet import SRNetModel, SRNetOptimized1f

        model_pos_train = SRNetOptimized1f(num_joints_in, in_features, num_joints_out,
                                                filter_widths=filter_widths, causal=args.causal, dropout=args.dropout,
                                                channels=args.channels,args=args)

        model_pos = SRNetModel(num_joints_in, in_features, num_joints_out,
                                    filter_widths = filter_widths, causal = args.causal, dropout = args.dropout, channels = args.channels, dense = args.dense, args=args)
        return model_pos_train, model_pos
        
    elif "PoseFormer" in args.model:
        from common.model_poseformer import PoseTransformer
        receptive_field = int(args.model.split('_')[1])
        model_pos_train = PoseTransformer(num_frame=receptive_field, num_joints=num_joints_in, in_chans=2, embed_dim_ratio=32, depth=4,
        num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,drop_path_rate=0.1)

        model_pos = PoseTransformer(num_frame=receptive_field, num_joints=num_joints_in, in_chans=2, embed_dim_ratio=32, depth=4,
        num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,drop_path_rate=0)
        return model_pos_train, model_pos

    elif args.model == "S4":
        from common.model_s4 import S4Model
        model_pos_train = S4Model(num_joints_in, in_features, num_joints_out, args.num_layers, dropout=args.dropout, channels=args.channels, 
                                    bidirectional=args.bidirectional)
        model_pos = S4Model(num_joints_in, in_features, num_joints_out, args.num_layers, dropout=args.dropout, channels=args.channels,
                                    bidirectional=args.bidirectional)
        return model_pos_train, model_pos  
    elif args.model == "S4Block":
        from common.model_s4 import S4ModelBlock
        model_pos_train = S4ModelBlock(num_joints_in, in_features, num_joints_out, dropout=args.dropout, channels=args.channels, 
                                    bidirectional=args.bidirectional, ff=2, architecture=args.architecture)
        model_pos = S4ModelBlock(num_joints_in, in_features, num_joints_out, dropout=args.dropout, channels=args.channels,
                                    bidirectional=args.bidirectional, ff=2, architecture=args.architecture)
        return model_pos_train, model_pos        
    else:
        raise NotImplementedError


def initialize_model(args, num_joints_in, num_joints_out, in_features, chk_filename=None):
    # Initialize train/test models
    model_pos_train, model_pos = get_model(args, 
        num_joints_in = num_joints_in, 
        num_joints_out= num_joints_out,
        in_features   = in_features)

    model_params = 0
    for parameter in model_pos.parameters():
        model_params += parameter.numel()
    print('INFO: Trainable parameter count:', model_params)

    # Move models to CUDA device(s)
    if torch.cuda.is_available():
        if args.parallel:
            model_pos       = nn.DataParallel(model_pos)
            model_pos_train = nn.DataParallel(model_pos_train)
        model_pos = model_pos.cuda()
        model_pos_train = model_pos_train.cuda()
    
    checkpoint = None
    # Load weights from pretrained models
    if args.resume or args.evaluate:
        print('Loading checkpoint', chk_filename)
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        print('This model was trained for {} epochs'.format(checkpoint['epoch']))
        model_pos_train.load_state_dict(checkpoint['model_pos'], strict=False)
        model_pos.load_state_dict(checkpoint['model_pos'])
    return model_pos, model_pos_train, checkpoint 