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
    
    elif args.model == "Attention3DHP":
        from common.model_attention_3dhp import Attention3DHPModelOptimized1f
        if not args.disable_optimizations and not args.dense and args.stride == 1:
            # Use optimized model for single-frame predictions
            model_pos_train = Attention3DHPModelOptimized1f(num_joints_in, in_features, num_joints_out,
                                                    filter_widths=filter_widths, causal=args.causal, dropout=args.dropout,
                                                    channels=args.channels)
        else:
            raise NotImplemented

        model_pos = Attention3DHPModelOptimized1f(num_joints_in, in_features, num_joints_out,
                                     filter_widths=filter_widths, causal=args.causal, dropout=args.dropout,
                                     channels=args.channels, dense=args.dense)
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
    # elif args.model == "ConfVideoPose3DV1":
    #     ModelClass  = ConfTemporalModelV1
    # elif args.model == "ConfVideoPose3DV2":
    #     ModelClass  = ConfTemporalModelV2
    # elif args.model == "ConfVideoPose3DV3":
    #     ModelClass  = ConfTemporalModelV3
    # elif args.model == "ConfVideoPose3DV31":
    #     ModelClass  = ConfTemporalModelV31
    # elif args.model == "ConfVideoPose3DV32":
    #     ModelClass  = ConfTemporalModelV32
    elif args.model == "ConfVideoPose3DV33":
        ModelClass  = ConfTemporalModelV33
    elif args.model == "ConfVideoPose3DV34":
        ModelClass  = ConfTemporalModelV34
    elif args.model == "ConfVideoPose3DV34gamma0":
        ModelClass  = ConfTemporalModelV34gamma0
    elif args.model == "ConfVideoPose3DV34gamma3":
        ModelClass  = ConfTemporalModelV34gamma3
    elif args.model == "ConfVideoPose3DV34gamma5":
        ModelClass  = ConfTemporalModelV34gamma5
    elif args.model == "ConfVideoPose3DV34tanh":
        ModelClass  = ConfTemporalModelV34tanh
    elif args.model == "ConfVideoPose3DV35":
        ModelClass  = ConfTemporalModelV35
    elif args.model == "ConfVideoPose3DV36":
        ModelClass  = ConfTemporalModelV36
    elif args.model == "ConfVideoPose3DV37":
        ModelClass  = ConfTemporalModelV37
    elif args.model == "ConfVideoPose3DV4":
        ModelClass  = ConfTemporalModelV4 
        
    else:
        raise NotImplementedError

    model_pos       = ModelClass(num_joints_in, in_features, num_joints_out,
                                filter_widths=filter_widths, causal=args.causal, dropout=args.dropout, channels=args.channels,
                                dense=args.dense)

    model_pos_train = ModelClass(num_joints_in, in_features, num_joints_out,
                                    filter_widths=filter_widths, causal=args.causal, dropout=args.dropout, channels=args.channels,
                                    dense=args.dense)
    return model_pos_train, model_pos

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