import argparse

def getAgrs():
    args = argparse.ArgumentParser()

    # CT scanning parameters
    args.add_argument('--nViews',type=int,default=360)
    args.add_argument('--nBins',type=int,default=600)
    args.add_argument('--resolution',type=int,default=512)

    # pre train parameters
    # filter moudle parameters
    args.add_argument('--f_isTrain',type=bool,default=False)
    args.add_argument('--f_gpu',type=list,default=[0])
    args.add_argument('--f_epoch',type=int,default=50000)
    args.add_argument('--f_lr',type=int,default=1e-4)
    args.add_argument('--f_isSaveModel',type=bool,default=False)
    args.add_argument('--f_isTest',type=bool, default=True)
    args.add_argument('--f_trainpath',type=str,default=r'E:\BRNData\train\projData_False')
    args.add_argument('--f_testpath',type=str,default=r'E:\BRNData\val\projData_False')
    args.add_argument('--f_savePath',type=str,default=r'/../filterOut/')

    # Extract block parameters
    args.add_argument('--e_isTrain',type=bool,default=False)
    args.add_argument('--e_block',type=int,default=36)
    args.add_argument('--e_bins',type=int,default=168)
    args.add_argument('--e_size',type=int,default=102)
    args.add_argument('--e_padding',type=int,default=160)
    args.add_argument('--e_proj_mask_file',type=str,default=r'./checkpoints/proj_mask.raw')
    args.add_argument('--e_image_mask_file',type=str,default=r'./checkpoints/image_mask.raw')
    args.add_argument('--e_isGenarateBlockdata',type=bool,default=True)
    args.add_argument('--e_gen_proj_path',type=str,default=r'E:\BRNData\train\filterOut')
    args.add_argument('--e_gen_image_path',type=str,default=r'E:\BRNData\train\label')
    args.add_argument('--e_save_gen_proj_path',type=str,default=r'E:\BRNData\train\block_proj')
    args.add_argument('--e_save_gen_image_path',type=str,default=r'E:\BRNData\train\block_image')
    args.add_argument('--e_test_gen_proj_path',type=str,default=r'E:\BRNData\val\filterOut')
    args.add_argument('--e_test_gen_image_path',type=str,default=r'E:\BRNData\val\label')
    args.add_argument('--e_test_save_gen_proj_path',type=str,default=r'E:\BRNData\val\block_proj')
    args.add_argument('--e_test_save_gen_image_path',type=str,default=r'E:\BRNData\val\block_image')
    args.add_argument('--e_shuffle',type=bool,default=False)
    args.add_argument('--e_num_block',type=int,default=1)
    args.add_argument('--e_lr',type=float,default=1e-4)
    args.add_argument('--e_batch_size',type=int,default=1)
    args.add_argument('--e_epoch',type=int,default=100)
    args.add_argument('--e_gpu',type=list,default=[0])
    args.add_argument('--e_save_bp_path',type=str,default=r'E:\BRNData\train\blockOut')
    args.add_argument('--e_test_save_bp_path',type=str,default=r'E:\BRNData\val\blockOut')

    # DNN of block back-projection module
    args.add_argument('--d_isTrain', type=bool, default=True)
    args.add_argument('--d_isMerge', type=bool, default=False)
    args.add_argument('--d_merge_mask',type=str,default=r'./checkpoints/merge_mask.raw')
    args.add_argument('--d_merge_train_path',type=str,default=r'E:\BRNData\train\mergedImage')
    args.add_argument('--d_merge_test_path',type=str,default=r'E:\BRNData\val\mergedImage')
    args.add_argument('--d_gpu',type=list,default=[0])
    args.add_argument('--d_batchsize',type=int,default=4)
    args.add_argument('--d_epoch',type=int,default=100)
    args.add_argument('--d_lr',type=float,default=1e-4)
    args.add_argument('--d_save_pretrain',type=str,default=r'E:\BRNData\train\dnnOut')
    args.add_argument('--d_test_save_pretrain',type=str,default=r'E:\BRNData\val\dnnOut')

    # train paramenters
    args.add_argument('--train_path',type=str,default=r'E:\BRNData\train\projData_False')
    args.add_argument('--label_path',type=str,default=r'E:\BRNData\train\label')
    args.add_argument('--epoch',type=int,default=100)
    args.add_argument('--lr',type=float,default=1e-7)
    args.add_argument('--batch_size',type=int,default=2)
    args.add_argument('--gpu',type=list,default=[0])


    return args.parse_args()