import numpy as np
class InputDistortion:
    def __init__(self, body_parts, type, temporal_distr = None, layout="h36m") -> None:
        self.body_parts = body_parts
        self.type  = type
        self.layout = layout 
        self.temporal_distr = temporal_distr 
    
    def get_mask(self):
        mask = np.ones(17) 
        if self.layout == "h36m":
            if self.body_parts == "legs":
                mask[1:7] = 0
            elif self.body_parts == "legs+root":
                mask[:7]  = 0
            elif self.body_parts == "arm-right":
                mask[14:17]=0
            elif self.body_parts == "arm-left" :
                mask[11:14]=0
            elif self.body_parts == "arms":
                mask[11:]  =0
            else:
                raise NotImplementedError
        elif self.layout == "coco":
            if self.body_parts == "legs":
                mask[11:] = 0
            else:
                raise NotImplementedError
        return mask

    def __apply_spatial_distortion(self, arr):
        msk = self.get_mask()
        if   self.type == 'none':
            return arr

        elif self.type == "zero":
            arr *= msk[:,None]

        elif self.type == "mean":
            arr[...,(1-msk).astype(bool),:] = arr[...,msk.astype(bool),:].mean(axis=1)[...,None,:]
            
        elif "gauss" in self.type:
            std = float(self.type.split('_')[1])
            arr[...,(1-msk).astype(bool),:] +=  np.random.normal(scale=std,size=arr[...,(1-msk).astype(bool),:].shape)
            
        elif self.type == "constant":
            arr[...,(1-msk).astype(bool),:] = arr[...,None,0,:]
        
        elif "joint" in self.type:
            joint_idx = int(self.type.split('_')[1])
            arr[...,(1-msk).astype(bool),:] = arr[...,None,joint_idx,:]
        
        else:
            raise NotImplementedError
        return arr
    
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

    def __call__(self, inputs):
        arr = np.copy(inputs)
        if self.temporal_distr == 'None' or None:
            arr = self.__apply_spatial_distortion(inputs) 
        else:
            ratio = float(self.temporal_distr)
            l     = arr.shape[1] if len(arr.shape)==4 else arr.shape[0]
            appl_idx  = self.__get_temporal_mask(l, ratio).astype(bool) 
            arr[appl_idx] = self.__apply_spatial_distortion(arr[appl_idx])
        return arr

    def remain_joints(self, outputs):
        msk = self.get_mask().astype(bool)
        return outputs[...,msk, :]