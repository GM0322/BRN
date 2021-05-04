import torch
import numpy as np
from utils import config
from dataset import dataset

def train():
    args = config.getAgrs()
    train_loader = dataset.getTrainDataLoader(args.train_path,args.label_path,args,args.batch_size,True)

    filterModule = torch.load(r'./checkpoints/FilterModule.pt')
    bpone = torch.load(r'./checkpoints/bpone.pt')
    dnn = torch.load(r'./checkpoints/dnn.pt')

    f_optim = torch.optim.SGD(filterModule.parameters(), lr=args.lr)
    e_optim = torch.optim.SGD(bpone.parameters(), lr=args.lr)
    d_optim = torch.optim.SGD(dnn.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()
    print('----------------start train BRN----------------------------')
    for epoch in range(args.d_epoch):
        for step,(x,y) in enumerate(train_loader):
            out = filterModule(x.cuda(args.gpu[0]))
            out = bpone(out.half())
            out = out[:,:,3:-3,3:-3].float()
            out = dnn(out)
            gt = y.view(y.size(0),1,102,102)[:,:,3:-3,3:-3].cuda(args.gpu[0])
            loss = criterion(out,gt)
            f_optim.zero_grad()
            e_optim.zero_grad()
            d_optim.zero_grad()
            loss.backward()
            f_optim.step()
            e_optim.step()
            d_optim.step()
            if(step%100 == 0):
                print('epoch = {},step={},loss = {}'.format(epoch,step,loss))
        out.data.cpu().numpy().astype(np.float32).tofile('1.raw')

if __name__ == '__main__':
    train()


