import torch

class FilterModule(torch.nn.Module):
    def __init__(self,kernel_size=21):
        super(FilterModule, self).__init__()
        self.conv1 = torch.nn.Conv2d( 1, 32, kernel_size=(1,kernel_size),padding=(0,kernel_size//2),bias=False)
        self.conv2 = torch.nn.Conv2d(32, 32, kernel_size=(1,kernel_size),padding=(0,kernel_size//2),bias=False)
        self.conv3 = torch.nn.Conv2d(32, 32, kernel_size=(1,kernel_size),padding=(0,kernel_size//2),bias=False)
        self.conv4 = torch.nn.Conv2d(32, 32, kernel_size=(1,kernel_size),padding=(0,kernel_size//2),bias=False)
        self.conv5 = torch.nn.Conv2d(32, 32, kernel_size=(1,kernel_size),padding=(0,kernel_size//2),bias=False)
        self.conv6 = torch.nn.Conv2d(32, 32, kernel_size=(1,kernel_size),padding=(0,kernel_size//2),bias=False)
        self.conv7 = torch.nn.Conv2d(32, 32, kernel_size=(1,kernel_size),padding=(0,kernel_size//2),bias=False)
        self.conv8 = torch.nn.Conv2d(32,  1, kernel_size=(1,kernel_size),padding=(0,kernel_size//2),bias=False)

    def forward(self, x):
        out = self.conv3(self.conv2(self.conv1(x)))
        out = self.conv6(self.conv5(self.conv4(out)))
        out = self.conv8(self.conv7(out))
        return out

class BlockReconMoudle(torch.nn.Module):
    def __init__(self, nBins=168, nViews=360, block_width=102):
        super(BlockReconMoudle, self).__init__()
        self.block_width = block_width
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(nBins * nViews, 10000, bias=False),
            torch.nn.Linear(10000, block_width * block_width, bias=False),
        )

    def forward(self,x):
        out = x.view(x.size(0)*x.size(1),-1)
        out = self.fc(out)
        out = out.view(out.size(0),1,self.block_width,self.block_width)
        return out

class ResBlock(torch.nn.Module):
    def __init__(self,channel):
        super(ResBlock, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(channel, channel, kernel_size=(3, 3), padding=(1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(channel, channel, kernel_size=(3, 3), padding=(1, 1)),
            torch.nn.ReLU()
        )

    def forward(self,input):
        return input+self.conv(input)

class ResUnet(torch.nn.Module):
    def __init__(self):
        super(ResUnet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1,64,kernel_size=(3,3),padding=(1,1))
        self.resBlock1 = ResBlock(64)
        self.conv2 = torch.nn.Conv2d(64,128,kernel_size=(3,3),padding=(1,1))
        self.resBlock2 = ResBlock(128)
        self.conv3 = torch.nn.Conv2d(128,256,kernel_size=(3,3),padding=(1,1))
        self.resBlock3 = ResBlock(256)
        self.conv4 = torch.nn.Conv2d(256,512,kernel_size=(3,3),padding=(1,1))
        self.resBlock4 = ResBlock(512)
        self.resBlock5 = ResBlock(512)
        self.conv5 = torch.nn.Conv2d(512, 256,kernel_size=(3,3),padding=(1,1))
        self.conv6_1 = torch.nn.Conv2d(512, 256, kernel_size=(3, 3), padding=(1, 1))
        self.resBlock6 = ResBlock(256)
        self.conv6_2 = torch.nn.Conv2d(256, 128, kernel_size=(3, 3), padding=(1, 1))
        self.conv7_1 = torch.nn.Conv2d(256, 128, kernel_size=(3, 3), padding=(1, 1))
        self.resBlock7 = ResBlock(128)
        self.conv7_2 = torch.nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1))
        self.conv8_1 = torch.nn.Conv2d(128,64,kernel_size=(3,3),padding=(1,1))
        self.resBlock8 = ResBlock(64)
        self.conv8_2 = torch.nn.Conv2d(64,1,kernel_size=(3,3),padding=(1,1))

        self.relu = torch.nn.ReLU()
        self.pooling = torch.nn.AvgPool2d(kernel_size=(2,2))
        self.upsampling = torch.nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self,input):
        x1 = self.resBlock1(self.relu(self.conv1(input)))                           #64 x 96 x 96
        x2 = self.resBlock2(self.relu(self.conv2(self.pooling(x1))))                #128x 48 x 48
        x3 = self.resBlock3(self.relu(self.conv3(self.pooling(x2))))                #256x 24 x 24
        x4 = self.resBlock4(self.relu(self.conv4(self.pooling(x3))))                #512x 12 x 12
        x5 = self.upsampling(self.relu(self.conv5(self.resBlock5(x4))))             #256x 24 x 24
        xup = self.relu(self.conv6_1(torch.cat((x3,x5),dim=1)))
        xup = self.upsampling(self.relu(self.conv6_2(self.resBlock6(xup))))         #128x 48 x 48
        xup = self.relu(self.conv7_1(torch.cat((x2,xup),dim=1)))
        xup = self.upsampling(self.relu(self.conv7_2(self.resBlock7(xup))))         #64 x 96 x 96
        xup = self.relu(self.conv8_1(torch.cat((x1,xup),dim=1)))
        xup = self.relu(self.conv8_2(self.resBlock8(xup)))
        return xup

