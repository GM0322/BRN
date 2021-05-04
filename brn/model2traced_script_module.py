import torch

filterModule = torch.load(r'./checkpoints/FilterModule.pt')
bpone = torch.load(r'./checkpoints/bpone.pt')
dnn = torch.load(r'./checkpoints/dnn.pt')

x = torch.randn(1,1,360,600,dtype=torch.float32).cuda()
filter_traced_script_module = torch.jit.trace(filterModule,x)
filter_traced_script_module.save('./checkpoints/c_FilterModule.pt')
x = torch.randn(36,1,360,168,dtype=torch.float16).cuda()
bp_traced_script_module = torch.jit.trace(bpone,x)
bp_traced_script_module.save('./checkpoints/c_bpone.pt')
x = torch.randn(1,1,512,512,dtype=torch.float32).cuda()
dnn_traced_script_module = torch.jit.trace(dnn,x)
dnn_traced_script_module.save("./checkpoints/c_dnn.pt")

