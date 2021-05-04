#pragma once
#include "torch/script.h"
#include "torch/torch.h"
//void recon(char* proj_path, int nViews, int nBins, float* recon_data, int resolution);


class BRN
{
public:
	void recon(char* proj_path, float* recon_data);
	BRN();

	~BRN();


private:
	int block;
	int block_size;
	int block_bins;
	int pad_size;
	int nViews;
	int nBins;
	int resolution;

	int* d_proj_mask;
	float* d_merge_mask;
	float* d_block_proj;
	float* d_filterData;
	int* d_image_index;


	torch::jit::script::Module filter_model;
	torch::jit::script::Module bpone_model;
	torch::jit::script::Module dnn_model;


};