import torch.utils.data as data
import os
import torch
import numpy as np
from utils import block

class filterDataset(data.Dataset):
    def __init__(self,input_path,nViews,nBins):
        self.input_path = input_path
        self.files = os.listdir(input_path)
        self.nBins = nBins
        self.nViews = nViews

    def __getitem__(self, item):
        input = torch.from_numpy(np.fromfile(self.input_path+'/'+self.files[item],
                                             dtype=np.float32)).view(1,self.nViews,self.nBins)
        return input,self.files[item]

    def __len__(self):
        return len(self.files)

def getFilterDataLoader(input_path,nViews,nBins):
    filterData = filterDataset(input_path,nViews,nBins)
    return data.DataLoader(dataset=filterData,batch_size=1,shuffle=False)

class blockProjectionDataset(data.Dataset):
    def __init__(self,args,input_path,label_path,num_block,shuffle):
        self.input_path = input_path
        self.label_path = label_path
        self.block = args.e_block
        self.view = args.nViews
        self.bins = args.e_bins
        self.block_size = args.e_size
        self.files = os.listdir(self.input_path)
        self.num_block= num_block
        self.shuffle = shuffle

    def __getitem__(self, item):
        input = torch.from_numpy(np.fromfile(self.input_path+'/'+self.files[item],dtype=np.float32)).view(self.block,self.view,self.bins)
        label = torch.from_numpy(np.fromfile(self.label_path+'/'+self.files[item],dtype=np.float32)).view(self.block,self.block_size,self.block_size)
        if self.shuffle:
            order = np.random.permutation(self.block)
        else:
            order = np.arange(0,self.block)
        return input[order[0:self.num_block],:,:],label[order[0:self.num_block],:,:],self.files[item]

    def __len__(self):
        return len(self.files)

def getBlockprojectionDataLoader(args,input_path,label_path,batch_size,num_block,shuffle):
    dataset = blockProjectionDataset(args,input_path,label_path,num_block,shuffle)
    return data.DataLoader(dataset,batch_size=batch_size,shuffle=shuffle)

class dnnDataset(data.Dataset):
    def __init__(self,input_path,label_path,size):
        self.input_path = input_path
        self.label_path = label_path
        self.files = os.listdir(input_path)
        self.size = size

    def __getitem__(self, item):
        input = torch.from_numpy(np.fromfile(self.input_path+'/'+self.files[item],dtype=np.float32)).view(1,self.size,self.size)
        label = torch.from_numpy(np.fromfile(self.label_path+'/'+self.files[item],dtype=np.float32))
        label = label.view(1,self.size,self.size).permute(0,2,1)
        return input,label,self.files[item]

    def __len__(self):
        return len(self.files)

def getDnnDataLoad(input_path,label_path,size,batch_size,shuffle):
    dataset = dnnDataset(input_path,label_path,size)
    return data.DataLoader(dataset,batch_size,shuffle=shuffle)

class trainDataset(data.Dataset):
    def __init__(self,input_path,label_path,args):
        self.args = args
        self.input_path = input_path
        self.label_path = label_path
        self.files = os.listdir(input_path)
        self.nViews = args.nViews
        self.nBins = args.nBins
        self.nSize = args.resolution
        self.mr = block.MaskRegion(args)

    def __getitem__(self, item):
        input = torch.from_numpy(np.fromfile(self.input_path+'/'+self.files[item],dtype=np.float32)).view(1,1,self.nViews,self.nBins)
        label = torch.from_numpy(np.fromfile(self.label_path+'/'+self.files[item],dtype=np.float32)).view(1,1,self.nSize,self.nSize)
        input = self.mr.image_padding(input)
        input,label = self.mr.getPair(input,label,np.random.permutation(self.args.e_block)[0])
        return input,label

    def __len__(self):
        return len(self.files)

def getTrainDataLoader(input_path,label_path,args,batch_size,shuffle):
    return data.DataLoader(trainDataset(input_path,label_path,args),batch_size=batch_size,shuffle=shuffle)