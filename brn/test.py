import torch
import numpy as np
from utils import config
from utils import block

# filterModule = module.FilterModule()
args = config.getAgrs()
input_data = torch.from_numpy(np.fromfile(r'.\test\input\001.raw',dtype=np.float32)).view(1,1,args.nViews,args.nBins)

filterModule = torch.load(r'./checkpoints/FilterModule.pt')
bpone = torch.load(r'./checkpoints/bpone.pt')
dnn = torch.load(r'./checkpoints/dnn.pt')
extract_block = block.MaskRegion(args)
merge_block = block.ImageMerge(args)

x = filterModule(input_data.cuda())
x = extract_block.getAllProj(x).view(args.e_block,1,args.nViews,-1)
y = bpone(x.half())
y = merge_block.merge(y.float().view(args.e_block,args.e_size,args.e_size))
out = dnn(y)
import matplotlib.pyplot as plt
res = (out[0,0,:,:].data.cpu().numpy()-0.1837)/0.1837*1000
plt.figure(1)
plt.imshow(res,cmap='gray',vmin=-160,vmax=240)
gt = np.fromfile(r'E:\BRNData\val\label\990.raw',dtype=np.float32).reshape(args.resolution,args.resolution)
plt.figure(2)
gt = (gt.T-0.1837)/0.1837*1000
plt.imshow(gt,cmap='gray',vmin=-160,vmax=240)
plt.figure(3)
plt.imshow(gt-res,cmap='gray',vmin=-50,vmax=50)
plt.show()









