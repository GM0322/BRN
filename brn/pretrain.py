from utils import config
from network import module
import torch
import numpy as np
from dataset import dataset
import os
from utils import block

def pre_train_filter_module(args):
    input = torch.eye(args.nBins,dtype=torch.float32).cuda(args.f_gpu[0]).view(1,1,args.nBins,args.nBins)
    label = torch.from_numpy(np.fromfile(r'./checkpoints/filter.raw',dtype=np.float32)).view(1,1,args.nBins,args.nBins).cuda(args.f_gpu[0])
    model = module.FilterModule().cuda(args.f_gpu[0])
    input = input[:,:,args.e_bins//2+1:args.nBins-args.e_bins//2-1,:]
    label = label[:,:,args.e_bins//2+1:args.nBins-args.e_bins//2-1,:]
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=args.f_lr)
    criterion = torch.nn.SmoothL1Loss()
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[i * args.f_epoch // 5  for i in range(1, 5)], gamma=0.55)
    for epoch in range(args.f_epoch):
        out = model(input)
        optim.zero_grad()
        loss = criterion(out, label)
        loss.backward()
        optim.step()
        lr_scheduler.step()
        if(epoch%1000 == 0):
            print('epoch = {},loss = {}'.format(epoch, loss))
    # out.data.cpu().numpy().tofile('./checkpoints/filterNNout.raw')
    model.eval()
    if(args.f_isSaveModel):
        torch.save(model, './checkpoints/FilterModule.pt')
    if(args.f_isTest):
        train_data = dataset.getFilterDataLoader(args.f_trainpath,args.nViews,args.nBins)
        if(os.path.isdir(args.f_trainpath+args.f_savePath) == False):
            os.mkdir(args.f_trainpath+args.f_savePath)
        for step,(x,fileName) in enumerate(train_data):
            out = model(x.cuda(args.f_gpu[0]))
            out.data.cpu().numpy().tofile(args.f_trainpath+args.f_savePath+fileName[0])
        test_data = dataset.getFilterDataLoader(args.f_testpath,args.nViews,args.nBins)
        if(os.path.isdir(args.f_testpath+args.f_savePath) == False):
            os.mkdir(args.f_testpath+args.f_savePath)
        for step,(x,fileName) in enumerate(test_data):
            out = model(x.cuda(args.f_gpu[0]))
            out.data.cpu().numpy().tofile(args.f_testpath+args.f_savePath+fileName[0])

def pre_train_bp_segment_one(args):
    if (args.e_isGenarateBlockdata):
        print('-----------------------------start generate pre-train data-----------------------------------------')
        mr = block.MaskRegion(args)
        mr.generatePretrainData(args.e_gen_proj_path, args.e_gen_image_path,
                                args.e_save_gen_proj_path,args.e_save_gen_image_path)
        mr.generatePretrainData(args.e_test_gen_proj_path, args.e_test_gen_image_path,
                                args.e_test_save_gen_proj_path,args.e_test_save_gen_image_path)
    train_loader = dataset.getBlockprojectionDataLoader(args,args.e_save_gen_proj_path,args.e_save_gen_image_path,
                                                        args.e_batch_size,args.e_num_block,shuffle=True)
    model = module.BlockReconMoudle().cuda(0).half()
    model = torch.load('./checkpoints/bpone.pt')
    optim = torch.optim.Adam(model.parameters(), lr=args.e_lr)
    criterion = torch.nn.SmoothL1Loss()
    print('-------------------------start pre-train segment one of bolck projection module--------------------')
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[i * args.f_epoch // 5  for i in range(1, 5)], gamma=0.55)
    for epoch in range(args.e_epoch):
        for step,(x,y,fileName) in enumerate(train_loader):
            out = model(x.cuda(args.e_gpu[0]).half())
            optim.zero_grad()
            loss = criterion(out,y.cuda(args.e_gpu[0]).half())
            loss.backward()
            optim.step()
            if(step%100 == 0):
                print('epoch = {},step={},loss = {}'.format(epoch,step,loss))
        lr_scheduler.step()

    print('------------------------------------------train finished------------------------------------------')
    print('-----------------start test and save the result of block backprojection module--------------------')
    model.eval()
    train_loader = dataset.getBlockprojectionDataLoader(args, args.e_save_gen_proj_path, args.e_save_gen_image_path, 1,
                                                        args.e_block, shuffle=False)
    test_loader = dataset.getBlockprojectionDataLoader(args, args.e_test_save_gen_proj_path, args.e_test_save_gen_image_path, 1,
                                                    args.e_block, shuffle=False)
    if(os.path.isdir(args.e_save_bp_path) == False):
        os.mkdir(args.e_save_bp_path)
    if(os.path.isdir(args.e_test_save_bp_path) == False):
        os.mkdir(args.e_test_save_bp_path)

    for step,(x,y,fileName) in enumerate(train_loader):
        out = model(x.cuda(args.e_gpu[0]).half())
        out.data.cpu().numpy().astype(np.float32).tofile(args.e_save_bp_path+'/'+fileName[0])
    for step, (x,y,fileName) in enumerate(test_loader):
        out = model(x.cuda(args.e_gpu[0]).half())
        out.data.cpu().numpy().astype(np.float32).tofile(args.e_test_save_bp_path+'/'+fileName[0])

def pre_train_dnn(args):
    if(args.d_isMerge):
        print('-----------------------------starting assmibling block image-------------------------------------')
        immerge = block.ImageMerge(args)
        immerge.fileMerge(args.e_save_bp_path,args.d_merge_train_path)
        immerge.fileMerge(args.e_test_save_bp_path,args.d_merge_test_path)

    print('-----------------------------starting trainning DNN-----------------------------------------------')
    model = module.ResUnet().cuda()
    train_loader = dataset.getDnnDataLoad(args.d_merge_train_path,args.e_gen_image_path,args.resolution,args.d_batchsize,True)
    optim = torch.optim.Adam(model.parameters(), lr=args.d_lr)
    criterion = torch.nn.SmoothL1Loss()
    print('----------------start pre-train segment two of bolck projection module----------------------------')
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[i * args.d_epoch // 5  for i in range(1, 5)], gamma=0.55)
    for epoch in range(args.d_epoch):
        for step,(x,y,fileName) in enumerate(train_loader):
            out = model(x.cuda(args.d_gpu[0]))
            optim.zero_grad()
            loss = criterion(out,y.cuda(args.d_gpu[0]))
            loss.backward()
            optim.step()
            if(step%100 == 0):
                print('epoch = {},step={},loss = {}'.format(epoch,step,loss))
        out.data.cpu().numpy().astype(np.float32).tofile('1.raw')
        lr_scheduler.step()

    model.eval()
    torch.save(model,'./checkpoints/dnn.pt')
    print('------------------------------------------train finished------------------------------------------')
    print('-----------------start test and save the result of block backprojection module--------------------')
    train_loader = dataset.getDnnDataLoad(args.d_merge_train_path,args.e_gen_image_path,args.resolution,1,False)
    test_loader = dataset.getDnnDataLoad(args.d_merge_test_path,args.e_test_gen_image_path,args.resolution,1,False)
    if(os.path.isdir(args.d_save_pretrain) == False):
        os.mkdir(args.d_save_pretrain)
    if(os.path.isdir(args.d_test_save_pretrain) == False):
        os.mkdir(args.d_test_save_pretrain)
    for step,(x,y,fileName) in enumerate(train_loader):
        out = model(x.cuda(args.d_gpu[0]))
        out.data.cpu().numpy().tofile(args.d_save_pretrain+'/'+fileName[0])
    for step,(x,y,fileName) in enumerate(test_loader):
        out = model(x.cuda(args.d_gpu[0]))
        out.data.cpu().numpy().tofile(args.d_test_save_pretrain+'/'+fileName[0])

def pre_train():
    args = config.getAgrs()
    if(args.f_isTrain):
        print('-------------------------------1.pre-train filter module------------------------------------------')
        pre_train_filter_module(args)
    if(args.e_isTrain):
        print('----------------2.pre-train segment one of block back-peojection module---------------------------')
        pre_train_bp_segment_one(args)
    if(args.d_isTrain):
        print('----------------3.pre-train segment two of block back-peojection module---------------------------')
        pre_train_dnn(args)

if __name__ == '__main__':
    pre_train()