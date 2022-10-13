from common.model  import *
from common.model_conf_scr import *
def get_model(args, num_joints_in, num_joints_out, in_features):
    filter_widths = [int(x) for x in args.architecture.split(',')]
    if args.model == "VideoPose3D":
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
    elif args.model == "PoseFormer":
        pass
    elif args.model == "ConfVideoPose3DV1":
        ModelClass  = ConfTemporalModelV1
    elif args.model == "ConfVideoPose3DV2":
        ModelClass  = ConfTemporalModelV2
    elif args.model == "ConfVideoPose3DV3":
        ModelClass  = ConfTemporalModelV3
    elif args.model == "ConfVideoPose3DV31":
        ModelClass  = ConfTemporalModelV31
    elif args.model == "ConfVideoPose3DV32":
        ModelClass  = ConfTemporalModelV32
    elif args.model == "ConfVideoPose3DV33":
        ModelClass  = ConfTemporalModelV33
    elif args.model == "ConfVideoPose3DV34":
        ModelClass  = ConfTemporalModelV34
    elif args.model == "ConfVideoPose3DV34sigmoid":
        ModelClass  = ConfTemporalModelV34sigmoid
    elif args.model == "ConfVideoPose3DV34tanh":
        ModelClass  = ConfTemporalModelV34tanh
    elif args.model == "ConfVideoPose3DV35":
        ModelClass  = ConfTemporalModelV35
    elif args.model == "ConfVideoPose3DV36":
        ModelClass  = ConfTemporalModelV36
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