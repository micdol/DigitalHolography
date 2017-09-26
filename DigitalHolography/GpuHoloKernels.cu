#include "GpuHoloKernels.cuh"
#include <stdio.h>

namespace gpu_holo {

	__host__ void getKernelDims(dim3 &blocks, dim3 &threads, int w, int h) {
		const int THREADS_X = 32;
		const int THREADS_Y = 32;
		const int BLOCKS_X = static_cast<int>(std::ceil(w) / static_cast<double>(THREADS_X));
		const int BLOCKS_Y = static_cast<int>(std::ceil(h) / static_cast<double>(THREADS_Y));
		threads = dim3(THREADS_X, THREADS_Y);
		blocks = dim3(BLOCKS_X, BLOCKS_Y);
	}

	__global__ void absSqKernel(
		const cv::cuda::PtrStepSz<double> re, 
		const cv::cuda::PtrStepSz<double> im, 
		cv::cuda::PtrStepSz<double> output)
	{
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x <= re.cols - 1 && y <= re.rows - 1 && y >= 0 && x >= 0)
		{
			double val_re = re(x, y);
			double val_im = im(x, y);
			output(x, y) = val_re * val_re + val_im * val_im;
		}
	}

	__host__ void callAbsSqKernel(
		const cv::cuda::PtrStepSz<double> re, 
		const cv::cuda::PtrStepSz<double> im, 
		cv::cuda::PtrStepSz<double> output)
	{
		dim3 blocks, threads;
		getKernelDims(blocks, threads, re.cols, re.rows);
		absSqKernel <<< blocks, threads >>> (re, im, output);
	}

	__global__ void argKernel(const cv::cuda::PtrStepSz<double> re, const cv::cuda::PtrStepSz<double> im, cv::cuda::PtrStepSz<double> output)
	{
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x <= re.cols - 1 && y <= re.rows - 1 && y >= 0 && x >= 0)
		{
			double val_re = re(x, y);
			double val_im = im(x, y);
			output(x, y) = atan2(val_im, val_re);
		}
	}
 
	__host__ void callArgKernel(const cv::cuda::PtrStepSz<double> re, const cv::cuda::PtrStepSz<double> im, cv::cuda::PtrStepSz<double> output)
	{
		dim3 blocks, threads;
		getKernelDims(blocks, threads, re.cols, re.rows);
		argKernel <<< blocks, threads >>> (re, im, output);
	}

	__global__ void logAbsKernel(const cv::cuda::PtrStepSz<double> re, const cv::cuda::PtrStepSz<double> im, cv::cuda::PtrStepSz<double> output)
	{
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x <= re.cols - 1 && y <= re.rows - 1 && y >= 0 && x >= 0)
		{
			double val_re = re(x, y);
			double val_im = im(x, y);
			output(x, y) = log(1.0 + val_re * val_re + val_im * val_im);
		}
	}

	__host__ void calllogAbsKernel(const cv::cuda::PtrStepSz<double> re, const cv::cuda::PtrStepSz<double> im, cv::cuda::PtrStepSz<double> output)
	{
		dim3 blocks, threads;
		getKernelDims(blocks, threads, re.cols, re.rows);
		logAbsKernel <<< blocks, threads >>> (re, im, output);
	}
}