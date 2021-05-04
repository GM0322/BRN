import astra
import numpy as np
import matplotlib.pyplot as plt

nView=360
nBins = 600
block = 6
image_size = 512
proj_bins = 168
block_size = 102
overlap = 10
padding = 160
geom = {'vol_geom': astra.create_vol_geom(image_size, image_size),
        'proj_geom': astra.create_proj_geom('parallel', 1.0 * image_size / nBins, nBins,
                                                    np.linspace(0, 2 * np.pi, 360, False))}

proj_id = astra.create_projector('cuda',geom['proj_geom'],geom['vol_geom'])
center = [overlap + (geom['vol_geom']['GridRowCount'] - overlap * 2) // (2 * block),
          overlap + (geom['vol_geom']['GridColCount'] - overlap * 2) // (2 * block)]

proj_index = np.zeros(shape=(block ** 2, nView), dtype=np.int32)
image_index = np.zeros(shape=(block ** 2, 2), dtype=np.int32)
for index in range(block**2):
    col = index//block
    row = index%block
    Mask = np.zeros((geom['vol_geom']['GridRowCount'],geom['vol_geom']['GridColCount']))
    image_index[index,0] = center[0]+row*(block_size-2*overlap)
    image_index[index,1] = center[1]+col*(block_size-2*overlap)
    Mask[image_index[index,0],image_index[index,1]] = 1
    sin_id, sinogram = astra.create_sino(Mask, proj_id)
    if(index == 0):
        plt.figure(1)
        plt.imshow(sinogram,cmap='gray')
    proj_index[index,:] = sinogram.argmax(axis=1)

proj_mask = np.zeros(shape=(block**2,nView,nBins+2*padding),dtype=np.int32)
image_mask = np.zeros(shape=(block**2,image_size,image_size),dtype=np.int32)
plt.figure(2)
for index in range(block**2):
    for j in range(nView):
        proj_mask[index,j,padding+proj_index[index,j]-proj_bins//2:padding+proj_index[index,j]+proj_bins//2] = 2
        proj_mask[index,j,padding+proj_index[index,j]:padding+proj_index[index,j]+1] = 4
    image_mask[index,image_index[index,0]-block_size//2:image_index[index,0]+block_size//2,
               image_index[index,1]-block_size//2:image_index[index,1]+block_size//2] = 2
    plt.subplot(6,6,index+1)
    plt.imshow(image_mask[index,:,:],cmap='gray')

proj_mask.tofile('../checkpoints/proj_mask.raw')
image_mask.tofile('../checkpoints/image_mask.raw')

plt.show()

