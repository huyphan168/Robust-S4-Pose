import torch
import numpy as np
class InputDistortion:
    def __init__(self, args) -> None:
        self.args   = args
        self.layout = args.dataset
        self.loss_ignore_parts = args.loss_ignore_parts
        self.eval_ignore_parts = args.eval_ignore_parts
        
    def get_mask(self, parts, out_shape= None):
        if out_shape != None:
            mask = np.ones(out_shape) # 1: keep, 0: distort 
        else:
            assert 'rand' not in self.loss_ignore_parts 
            mask = np.ones((1,
                17 if self.layout == 'h36m' else 15,
            1))
        if self.layout == "h36m":
            if   parts == "legs":
                mask[:,1:7,:] = 0
            elif parts == "legs+root":
                mask[:,:7,:]  = 0
            elif parts == "arm-right":
                mask[:,14:17,:]=0
            elif parts == "arm-left" :
                mask[:,11:14,:]=0
            elif parts == "arms":
                mask[:,11:,:]  =0
            elif parts == "all": 
                mask   = np.zeros(out_shape)
            elif parts == "None":
                pass
            else:
                raise NotImplementedError
        elif self.layout == "coco":
            if parts == "legs":
                mask[11:] = 0
            else:
                raise NotImplementedError
        
        if 'randfix' in parts:
            p = float(parts.split('_')[1])
            mask[:,np.random.rand(out_shape[1]) < p,:] = 0
        elif 'rand' in parts:
            p = float(parts.split('_')[1])
            mask[np.random.rand(*out_shape[:-1]) <p, :] = 0
        
        if out_shape != None:
            return mask
        else:
            return mask[0,:,0]

    def __apply_spatial_distortion(self, arr, body_parts, type):
        msk = self.get_mask(body_parts, arr.shape)
        added_noise = np.zeros_like(arr)
        if   type == 'None':
            pass
        elif type == "zero":
            arr *= msk

        elif type == "mean":
            arr[(1-msk).astype(bool)] = arr[msk.astype(bool)].mean(axis=1)[...,None,:]
            
        elif "gauss" in type:
            std = float(type.split('_')[1])
            added_noise[(1-msk).astype(bool)] =  np.random.normal(scale=std,size=added_noise[(1-msk).astype(bool)].shape)
            arr += added_noise
        
        elif "laplace" in type:
            scl = float(type.split('_')[1])
            added_noise[(1-msk).astype(bool)] =  np.random.laplace(scale=scl,size=added_noise[(1-msk).astype(bool)].shape)
            arr += added_noise

        elif "impulse" in type:
            q   = float(type.split('_')[1])
            impl_msk = np.random.rand(*msk.shape) < q
            impl_msk = np.logical_and(impl_msk, (1-msk).astype(bool))
            arr[impl_msk] = np.where(np.random.rand(*arr[impl_msk].shape) <0.3, -0.5, 0.5)
            
        elif type == "constant":
            arr[(1-msk).astype(bool)] = arr[...,None,0,:]
        
        elif type == "exclude":
            arr = arr[(msk).astype(bool)]

        elif "joint" in type:
            joint_idx = int(type.split('_')[1])
            arr[(1-msk).astype(bool)] = arr[...,None,joint_idx,:]
        
        else:
            raise NotImplementedError
        return arr, msk, added_noise
    
    def __get_temporal_mask(self, l, ratio):
        len_keep = int(l * (1 - ratio))
        noise = np.random.rand(l)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = np.argsort(noise)  # ascend: small is keep, large is removed
        ids_restore = np.argsort(ids_shuffle)
        # generate the binary mask: 0 is kept, 1 is removed
        mask = np.ones(l)
        mask[:len_keep] = 0
        # unshuffle to get the binary mask
        mask = np.take(mask, indices=ids_restore)
        return mask

    def __apply_input_distortion(self, inputs, body_parts, type, temporal_distr, gen_conf_score):
        assert len(inputs.shape) == 3 
        f, j, d = inputs.shape
        arr = np.copy(inputs)
        if self.args.drop_conf_score == True:
            arr = arr[...,:-1]

        if temporal_distr != 'None':
            ratio = float(temporal_distr)
            appl_msk  = self.__get_temporal_mask(f, ratio).astype(bool) 
        else:
            appl_msk  = np.ones(f).astype(bool)
        
        arr[appl_msk], msk, added_noise = self.__apply_spatial_distortion(np.copy(arr[appl_msk,:,:]), body_parts, type)
        
        if gen_conf_score != 'None':
            conf_scr = np.ones((f,j,1))
            if gen_conf_score == 'binary':
                conf_scr[appl_msk] = msk[...,:1]
            elif gen_conf_score == 'gauss':
                assert "gauss" in type
                std = float(type.split('_')[1])
                conf_scr[appl_msk] = np.exp(-1/2 * np.sum((added_noise/std) ** 2, axis=-1))[...,None]
            arr = np.concatenate([arr, conf_scr], axis=-1)
        return arr

    def get_train_inputs(self, inputs):
        tmp = self.__apply_input_distortion(inputs, 
            body_parts= self.args.train_distortion_parts,
            type      = self.args.train_distortion_type,
            temporal_distr= self.args.train_distortion_temporal,
            gen_conf_score= self.args.train_gen_conf_score    
        )
        return tmp

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

    # def smooth_conf_scr(self, inputs):
    #     if self.args.smooth_conf_score == True:
    #         assert inputs.shape[-1] == 3
    #         inputs[...,-1] = torch.softmax(inputs[...,-1], dim=1)
    #     return inputs