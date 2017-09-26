#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

namespace gpu_holo {

	__host__ void getKernelDims(dim3 &blocks, dim3 &threads, int w, int h);
	 
	__global__ void absSqKernel(
		const cv::cuda::PtrStepSz<double> re, 
		const cv::cuda::PtrStepSz<double> im,
		cv::cuda::PtrStepSz<double> output);
	__host__ void callAbsSqKernel(
		const cv::cuda::PtrStepSz<double> re,
		const cv::cuda::PtrStepSz<double> im,
		cv::cuda::PtrStepSz<double> output);

	__global__ void argKernel(
		const cv::cuda::PtrStepSz<double> re,
		const cv::cuda::PtrStepSz<double> im,
		cv::cuda::PtrStepSz<double> output);
	__host__ void callArgKernel(
		const cv::cuda::PtrStepSz<double> re,
		const cv::cuda::PtrStepSz<double> im,
		cv::cuda::PtrStepSz<double> output);

	__global__ void logAbsKernel(
		const cv::cuda::PtrStepSz<double> re,
		const cv::cuda::PtrStepSz<double> im,
		cv::cuda::PtrStepSz<double> output);
	__host__ void calllogAbsKernel(
		const cv::cuda::PtrStepSz<double> re,
		const cv::cuda::PtrStepSz<double> im,
		cv::cuda::PtrStepSz<double> output);
}