import numpy as np
class InputDistortion:
    def __init__(self, body_parts, type, temporal_distr = None, layout="h36m") -> None:
        self.body_parts = body_parts
        self.type  = type
        self.layout = layout 
        self.temporal_distr = temporal_distr 
    
    def get_mask(self):
        mask = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]) 
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

    def __call__(self, inputs):
        msk = self.get_mask()
        import ipdb; ipdb.set_trace()
        arr = np.copy(inputs)
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
    
    def remain_joints(self, outputs):
        msk = self.get_mask().astype(bool)
        return outputs[...,msk, :]