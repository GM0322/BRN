#include "recon.h"

#include "opencv2/opencv.hpp"
#include "cuda_runtime.h"

#define kBLOCK 8

#define cudaCheckErrors(msg) \
do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) {    printf("%s\n",msg);  } \
} while (0)


__global__ void getProjBlockkernel(int* proj_mask,float* proj_data,float* block_proj,
	int block,int nViews,int nBins,int block_bins)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x > block||y>nViews)
		return;
	
	int index = 0;
	for (int i = 0; i < nBins; ++i)
	{
		if (proj_mask[x * nViews * nBins + y * nBins + i] > 1)
		{
			block_proj[x * nViews * block_bins + y * block_bins + index] = proj_data[y * nBins + i];
			++index;
		}
		
	}
	//printf("%3d,%3d,%3d,%3d,%3d,%3d,%3d\n", x, y, block, nViews, nBins, block_bins,index);
	//printf("%d\n", index);
}

__global__ void paddingkernel(float* proj_data,float* pad_data, int height, int width, int pad)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x > width || y > height)return;
	//printf("%d.%d\n", height, width);
	pad_data[y * (width+2*pad) + x + pad] = proj_data[y * width + x];
}

__global__ void getMergeImagekernel(float* merge_mask,int* image_index,float*block_image, float* merged_image,
	int block,int image_size, int block_size)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x > image_size || y > image_size)return;
	merged_image[y * image_size + x] = 0.0;
	for (int i = 0; i < block; ++i)
	{
		int b_x = x - image_index[2 * i];
		int b_y = y - image_index[2 * i + 1];
		if (b_x < 0 || b_x >= block_size || b_y < 0 || b_y >= block_size) continue;
		merged_image[y * image_size + x] += merge_mask[i * image_size * image_size + y * image_size + x] * block_image[i * block_size * block_size + b_y * block_size + b_x];
	}
}

BRN::BRN() {
	block = 36;
	block_size = 102;
	block_bins = 168;

	pad_size = 160;
	nViews = 360;
	nBins = 600;
	resolution = 512;
	int* proj_mask = new int[size_t(block) * nViews * (nBins + 2 * pad_size)];
	float* merge_mask = new float[size_t(block) * resolution * resolution];
	FILE* fp;
	// load mask file
	fopen_s(&fp, "proj_mask.raw", "rb");
	fread(proj_mask, 1, sizeof(int) * size_t(nViews) * block * (nBins + 2 * pad_size), fp);
	fclose(fp);
	fopen_s(&fp, "merge_mask.raw", "rb");
	fread(merge_mask, 1, sizeof(float) * block * resolution * resolution, fp);
	fclose(fp);
	
	cudaMalloc((void**)&d_filterData, sizeof(float) * nViews * (nBins + 2 * pad_size));
	cudaMalloc((void**)&d_proj_mask, sizeof(int) * block * nViews * (nBins + 2 * pad_size));
	cudaMalloc((void**)&d_merge_mask, sizeof(float) * block * resolution * resolution);
	cudaMalloc((void**)&d_block_proj, sizeof(float) * block * nViews * block_bins);

	cudaMemcpy(d_proj_mask, proj_mask, sizeof(int) * block * nViews * (nBins + 2 * pad_size), cudaMemcpyHostToDevice);
	cudaMemcpy(d_merge_mask, merge_mask, sizeof(float) * block * resolution * resolution, cudaMemcpyHostToDevice);
	cudaMemset(d_block_proj, 0, sizeof(float) * block * nViews * block_bins);

	// load each module
	try { filter_model = torch::jit::load("c_FilterModule.pt"); }
	catch (const c10::Error& e) {
		std::cerr << "error loading filter model\n";
		getchar();
		return;
	}
	try { bpone_model = torch::jit::load("c_bpone.pt"); }
	catch (const c10::Error& e) {
		std::cerr << "error loading the segment one of block backprojection module\n";
		getchar();
		return;
	}
	try { dnn_model = torch::jit::load("c_dnn.pt"); }
	catch (const c10::Error& e) {
		std::cerr << "error loading the segment two of block backprojection module\n";
		getchar();
		return;
	}
	int* index = new int[6];
	index[0] = 0; index[1] = 133 - 51; index[2] = 215 - 51;
	index[3] = 297 - 51; index[4] = 379 - 51; index[5] = 461 - 51;
	int* image_index = new int[block*2];
	for (size_t i = 0; i < 6; i++)
	{
		for (size_t j = 0; j < 6; j++)
		{
			image_index[2 * (i * 6 + j)] = index[j];
			image_index[2 * (i * 6 + j) + 1] = index[i];
		}
	}
	cudaMalloc((void**)&d_image_index, sizeof(int) * block * 2);
	cudaMemcpy(d_image_index, image_index, sizeof(int) * block * 2, cudaMemcpyHostToDevice);

	delete image_index;
	delete proj_mask;
	delete merge_mask;

}

BRN::~BRN()
{
	cudaFree(d_proj_mask);
	cudaFree(d_merge_mask);
	cudaFree(d_block_proj);
	cudaFree(d_filterData);
	cudaFree(d_image_index);
}



void BRN::recon(char* proj_path,float* recon_data)
{
    if (!torch::cuda::is_available() || !torch::cuda::cudnn_is_available())
        return;	
	// read projection data	
	float* proj_data = new float[size_t(nViews) * nBins];
	FILE* fp;
	fopen_s(&fp, proj_path, "rb");
	fread(proj_data, 1, sizeof(float) * nViews * nBins, fp);
	fclose(fp);
	torch::Tensor tensor_img = torch::from_blob(proj_data, { 1, 1, nViews, nBins }, torch::kFloat32).cuda();
	std::vector<torch::jit::IValue> inputs;
	std::vector<torch::jit::IValue> bpOneinput;
	std::vector<torch::jit::IValue> dnninput;

	dim3 bPadDim(kBLOCK, kBLOCK);
	dim3 gPadDim((nBins + kBLOCK - 1) / kBLOCK, (nViews + kBLOCK - 1) / kBLOCK);
	dim3 bBlockDim(kBLOCK, kBLOCK);
	dim3 gBlockDim((block + kBLOCK - 1) / kBLOCK, (nViews + kBLOCK - 1) / kBLOCK);
	dim3 bMergeDim(kBLOCK, kBLOCK);
	dim3 gMergeDim((resolution + kBLOCK - 1) / kBLOCK, (resolution + kBLOCK - 1) / kBLOCK);

	// forward processing for reconstruction
	cudaError_t __err;
	inputs.push_back(tensor_img);
	at::Tensor filter_res = filter_model.forward(inputs).toTensor();
	cudaMemset(d_filterData, 0, sizeof(float) * nViews * (nBins + 2 * pad_size));
	paddingkernel << <gPadDim, bPadDim >> > ((float*)filter_res.data_ptr(), d_filterData, nViews, nBins, pad_size);	
	torch::Tensor bp_input = torch::zeros({ block,1,nViews,block_bins }, torch::kFloat32).cuda();	
	getProjBlockkernel << <gBlockDim, bBlockDim >> > (d_proj_mask, d_filterData, (float*)bp_input.data_ptr(), block, nViews, nBins + 2 * pad_size, block_bins);
	bp_input = bp_input.toType(torch::kFloat16);
	bpOneinput.push_back(bp_input);
	at::Tensor bpone_res = bpone_model.forward(bpOneinput).toTensor();
	bpone_res = bpone_res.toType(torch::kFloat32);
		
	torch::Tensor dnn_input = torch::zeros({ 1,1,resolution,resolution }, torch::kFloat32).cuda();
	getMergeImagekernel <<<gMergeDim, bMergeDim >>> (d_merge_mask, d_image_index, (float*)bpone_res.data_ptr(), (float*)dnn_input.data_ptr(), block, resolution, block_size);
	dnninput.push_back(dnn_input);
	at::Tensor dnn_res = dnn_model.forward(dnninput).toTensor();
	cudaMemcpy(recon_data, (float*)dnn_res.data_ptr(), sizeof(float) * resolution * resolution, cudaMemcpyDeviceToHost);
}