import imp
import numpy as np
class InputDistortion:
    def __init__(self, args, layout="h36m") -> None:
        self.args   = args
        self.layout = layout
        self.loss_ignore_parts = args.loss_ignore_parts
        self.eval_ignore_parts = args.eval_ignore_parts

    def get_mask(self, parts):
        mask = np.ones(17) # 1: keep, 0: distort 
        if self.layout == "h36m":
            if   parts == "legs":
                mask[1:7] = 0
            elif parts == "legs+root":
                mask[:7]  = 0
            elif parts == "arm-right":
                mask[14:17]=0
            elif parts == "arm-left" :
                mask[11:14]=0
            elif parts == "arms":
                mask[11:]  =0
            elif parts == "all": 
                mask[:]    =0
            elif parts == "None":
                pass
            elif 'rand' in parts:
                p = float(parts.split('_')[1])
                mask = (np.random.rand(17) < p).astype(np.float)
            else:
                raise NotImplementedError
        elif self.layout == "coco":
            if parts == "legs":
                mask[11:] = 0
            else:
                raise NotImplementedError
        return mask

    def __apply_spatial_distortion(self, arr, body_parts, type):
        msk = self.get_mask(body_parts)
        added_noise = np.zeros_like(arr)
        if   type == 'None':
            pass
        elif type == "zero":
            arr *= msk[:,None]

        elif type == "mean":
            arr[...,(1-msk).astype(bool),:] = arr[...,msk.astype(bool),:].mean(axis=1)[...,None,:]
            
        elif "gauss" in type:
            std = float(type.split('_')[1])
            added_noise[...,(1-msk).astype(bool),:] =  np.random.normal(scale=std,size=added_noise[...,(1-msk).astype(bool),:].shape)
            arr += added_noise
        
        elif type == "constant":
            arr[...,(1-msk).astype(bool),:] = arr[...,None,0,:]
        
        elif type == "exclude":
            arr = arr[...,(msk).astype(bool),:]

        elif "joint" in type:
            joint_idx = int(type.split('_')[1])
            arr[...,(1-msk).astype(bool),:] = arr[...,None,joint_idx,:]
        
        else:
            raise NotImplementedError
        return arr, msk, added_noise
    
    def __get_temporal_mask(self, l, ratio):
        len_keep = int(l * (1 - ratio))
        noise = np.random.rand(l)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = np.argsort(noise)  # ascend: small is keep, large is removed
        ids_restore = np.argsort(ids_shuffle)
        # generate the binary mask: 0 is keep, 1 is remove
        mask = np.ones(l)
        mask[:len_keep] = 0
        # unshuffle to get the binary mask
        mask = np.take(mask, indices=ids_restore)
        return mask

    def __apply_input_distortion(self, inputs, body_parts, type, temporal_distr, gen_conf_score):
        assert len(inputs.shape) == 3 
        f, j, d = inputs.shape
        # l     = inputs.shape[1] if len(arr.shape)==4 else inputs.shape[0]
        if temporal_distr != 'None':
            ratio = float(temporal_distr)
            appl_msk  = self.__get_temporal_mask(f, ratio).astype(bool) 
        else:
            appl_msk  = np.ones(f).astype(bool)
        
        arr = np.copy(inputs)
        arr[appl_msk], msk, added_noise = self.__apply_spatial_distortion(np.copy(inputs[appl_msk,:,:]), body_parts, type)
        
        if gen_conf_score != 'None':
            conf_scr = np.ones((f,j,1))
            if gen_conf_score == 'binary':
                conf_scr[appl_msk] = msk[None,:,None]
            elif gen_conf_score == 'gauss':
                assert "gauss" in type
                std = float(type.split('_')[1])
                conf_scr[appl_msk] = np.exp(-1/2 * np.sum((added_noise/std) ** 2, axis=-1))[...,None]
            arr = np.concatenate([arr, conf_scr], axis=-1)
        return arr

    def get_train_inputs(self, inputs):
        return self.__apply_input_distortion(inputs, 
            body_parts= self.args.train_distortion_parts,
            type      = self.args.train_distortion_type,
            temporal_distr= self.args.train_distortion_temporal,
            gen_conf_score= self.args.train_gen_conf_score    
        )

    def get_test_inputs(self, inputs):
        return self.__apply_input_distortion(inputs, 
            body_parts= self.args.test_distortion_parts,
            type      = self.args.test_distortion_type,
            temporal_distr= self.args.test_distortion_temporal,
            gen_conf_score= self.args.test_gen_conf_score    
        )

    def get_loss_joints(self, inputs):
        return inputs[...,self.get_mask(self.loss_ignore_parts).astype(bool), :]

    def get_eval_joints(self, inputs):
        return inputs[...,self.get_mask(self.eval_ignore_parts).astype(bool), :]