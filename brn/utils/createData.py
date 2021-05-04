import astra
import numpy as np
import os
class DatasetParallelGenerate():
    def __init__(self, gtPath, sPath, args,isNoisy=False,isRecon=True):
        self.vol_geom = astra.create_vol_geom(args.resolution, args.resolution),
        self.proj_geom = astra.create_proj_geom('parallel', 1.0 * args.resolution / args.nBins, args.nBins,np.linspace(0, 2 * np.pi, 360, False)),
        self.gtPath = gtPath
        self.sPath = sPath
        self.isNoisy = isNoisy
        self.files = os.listdir(self.gtPath)
        self.proj_id = astra.create_projector('cuda',self.proj_geom,self.vol_geom)
        self.nBins = self.proj_geom['DetectorCount']
        self.isRecon = isRecon

    def projection(self, ImageData):
        sin_id,sinogram = astra.create_sino(ImageData,self.proj_id)
        if self.isNoisy:
            I0 = 5e4
            maxv = sinogram.max()
            counts = I0*np.exp(-sinogram/maxv)
            noisy_counts = np.random.poisson(counts)
            sinogram = -np.log(noisy_counts/I0)*maxv
            astra.data2d.store(sin_id,sinogram)
        rec = None
        if self.isRecon:
            rec_id = astra.data2d.create('-vol', self.vol_geom)
            cfg = astra.astra_dict('FBP_CUDA')
            cfg['ReconstructionDataId'] = rec_id
            cfg['ProjectionDataId'] = sin_id
            cfg['option'] = {'FilterType': 'shepp-logan'}
            alg_id = astra.algorithm.create(cfg)
            astra.algorithm.run(alg_id)
            rec = astra.data2d.get(rec_id)
            astra.algorithm.delete(alg_id)
            astra.data2d.delete(rec_id)
            astra.data2d.delete(sin_id)
            astra.projector.delete(rec_id)
        sinogram = sinogram.astype(np.float32)
        return sinogram,rec/(self.proj_geom['DetectorWidth']**2)

    def Generate(self):
        if os.path.isdir(self.sPath+'_'+str(self.isNoisy)) == False:
            os.mkdir(self.sPath+'_'+str(self.isNoisy))
        if os.path.isdir(self.sPath+'/../FBP_'+str(self.isNoisy)) == False:
            os.mkdir(self.sPath+'/../FBP_'+str(self.isNoisy))
        for i, file in enumerate(self.files):
            I = np.fromfile(self.gtPath+'/'+file,dtype=np.float32).reshape(self.vol_geom['GridRowCount'],self.vol_geom['GridColCount'])
            I[I<0] = 0
            sinogram,recon = self.projection(I)
            sinogram.tofile(self.sPath+'_'+str(self.isNoisy)+'/'+file)
            if self.isRecon:
                recon.tofile(self.sPath+'/../FBP_'+str(self.isNoisy)+'/'+file)