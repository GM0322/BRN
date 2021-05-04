import torch
import numpy as np
import os
from numba import jit
from utils import MutilThread

class MaskRegion():
    def __init__(self,args):
        self.bins = args.nBins
        self.view = args.nViews
        self.image_size=args.resolution
        self.block = args.e_block
        self.block_bins = args.e_bins
        self.block_size = args.e_size
        self.padding = args.e_padding
        self.proj_mask = np.fromfile(args.e_proj_mask_file,dtype=np.int32).reshape(self.block,self.view,self.bins+2*self.padding)
        self.image_mask = np.fromfile(args.e_image_mask_file,dtype=np.int32).reshape(self.block,self.image_size,self.image_size)
        self.conv = torch.nn.Conv2d(1,1,kernel_size=(1,1),padding=(0,self.padding),bias=False)
        self.conv.weight.data[0] = 1.0

    def getPair(self,proj, image, index):
        block_proj = proj[:,:,self.proj_mask[index,:,:]>1]
        block_image = image[:,:,self.image_mask[index,:,:]>1]
        return block_proj.to(proj.device),block_image.to(image.device)

    def getAllProj(self,proj):
        proj_data = self.image_padding(proj)
        block_proj = torch.zeros((self.block, self.view * self.block_bins), dtype=torch.float32).to(proj.device)
        for i in range(self.block):
            block_proj[i, :] = proj_data[:,:,self.proj_mask[i,:,:]>1]
        return block_proj

    def getOneProj(self,proj_data,index):
        return proj_data[:,:,self.proj_mask[index,:,:]>1]

    def getAllProjMultiProcessing(self,proj):
        proj_data = self.image_padding(proj)
        block_proj = torch.zeros((self.block, self.view * self.block_bins), dtype=torch.float32).to(proj.device)
        for i in range(self.block):
            task = MutilThread.MyThread(self.getOneProj,(proj_data,i))
            task.start()
            task.join()
            block_proj[i, :] = task.get_result()
        return block_proj

    def image_padding(self,proj):
        self.conv = self.conv.to(proj.device)
        return self.conv(proj)

    def generatePretrainData(self,proj_path,image_path,sproj_path,simage_path):
        files = os.listdir(proj_path)
        if(os.path.isdir(sproj_path) == False):
            os.mkdir(sproj_path)
        if(os.path.isdir(simage_path) == False):
            os.mkdir(simage_path)
        for file in files:
            proj_data = torch.from_numpy(np.fromfile(proj_path+'/'+file,dtype=np.float32)).view(1,1,self.view,self.bins)
            proj_data = self.image_padding(proj_data)
            image_data = torch.from_numpy(np.fromfile(image_path+'/'+file,dtype=np.float32)).view(1,1,self.image_size,self.image_size)
            block_proj = torch.zeros((self.block,self.view*self.block_bins),dtype=torch.float32)
            block_image = torch.zeros((self.block,self.block_size*self.block_size),dtype=torch.float32)
            for i in range(self.block):
                block_proj[i,:],block_image[i,:] = self.getPair(proj_data,image_data,i)
            block_image = block_image.view(self.block,self.block_size,self.block_size).permute(0,2,1)
            block_proj.data.numpy().tofile(sproj_path+'/'+file)
            block_image.data.numpy().tofile(simage_path+'/'+file)

class ImageMerge():
    def __init__(self,args):
        self.block = args.e_block
        self.size = args.resolution
        self.block_size = args.e_size
        self.mask = np.fromfile(args.d_merge_mask,dtype=np.float32).reshape(args.e_block,args.resolution*args.resolution)
        self.tensor_mask = torch.from_numpy(self.mask)

    def merge(self,block_image):
        data = torch.zeros((self.block,self.size*self.size),dtype=torch.float32).to(block_image.device)
        image = torch.zeros((self.size*self.size),dtype=torch.float32).to(block_image.device)
        self.tensor_mask = self.tensor_mask.to(block_image.device)
        for i in range(self.mask.shape[0]):
            data[i,self.mask[i,:]>1e-5] = block_image[i,:,:].view(-1)#.view(self.block_size,self.block_size)
            image = image+data[i]*self.tensor_mask[i,:]
        return image.view(1,1,self.size,self.size)

    def fileMerge(self,input_path,save_path):
        files = os.listdir(input_path)
        if(os.path.isdir(save_path) == False):
            os.mkdir(save_path)
        for i in files:
            input_data = torch.from_numpy(np.fromfile(input_path+'/'+i,dtype=np.float32)).view(self.block,self.block_size,self.block_size)
            save_data = self.merge(input_data)
            save_data.numpy().tofile(save_path+'/'+i)
