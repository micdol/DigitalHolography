#include <chrono>

#include "CpuHolo.hpp"
#include "GpuHolo.hpp"

using namespace cv;
using namespace cv::cuda;

const cv::String dir = "D:\\Studies\\PW\\MASTERS\\Images\\";
const cv::String lena = dir + "lena1024.jpg";
const cv::String baboon = dir + "baboon1024.jpg";
const cv::String holo = dir + "holo_3456.jpg";
const cv::String noise = dir + "noise.jpg";

void benchmark() {
	const int TIMES = 100;
	auto t_start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < TIMES; i++) {

	}
	auto t_end = std::chrono::high_resolution_clock::now();
	auto total = std::chrono::duration<double, std::milli>(t_end - t_start).count();
	cout << std::fixed << std::setprecision(2)
		<< "Total: " << total << "ms" << endl
		<< "Per run: " << total / TIMES << "ms" << endl;

} 

void run_cpu_tests() {
	CpuHolo c1 = CpuHolo(lena, baboon);
	c1.info();
	c1.show();

	c1.adjustROI(50, 50, 250, 250);
	c1.info();
	c1.show();

	c1.highPassFilter(100);
	c1.show();

	/*
	const int TIMES = 100;
	auto t_start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < TIMES; i++) {
		c1.mulTransferFunction(0.66);
	}
	auto t_end = std::chrono::high_resolution_clock::now();
	auto total = std::chrono::duration<double, std::milli>(t_end - t_start).count();
	cout << std::fixed << std::setprecision(2)
		<< "Total: " << total << "ms" << endl
		<< "Per run: " << total / TIMES << "ms" << endl;
	*/

	c1.expand(200);
	c1.info();
	c1.show();

	c1.expand(512);
	c1.info();
	c1.show();

	namedWindow("roi", CV_WINDOW_NORMAL);
	imshow("roi", c1.roi());
	waitKey(0);

	c1 = CpuHolo(lena, baboon);
	c1.info();
	c1.show();

	c1.expand(512, 768);
	c1.info();
	c1.show();

	c1.expand(1536, 768);
	c1.info();
	c1.show();

	c1 = CpuHolo(lena, 512, 512);
	c1.info();
	c1.show();

	c1 = CpuHolo(baboon, lena, 1536, 1536);
	c1.info();
	c1.show();

	imshow("roi", c1.roi());
	waitKey(0);
}

void run_gpu_tests() {
	GpuHolo c1 = GpuHolo(lena);
	c1.info();
	c1.show();

	/*
	c1.adjustROI(50, 50, 250, 250);
	c1.info();
	c1.show();

	c1.highPassFilter(100);
	c1.show();

	c1.expand(200);
	c1.info();
	c1.show();

	c1.expand(512);
	c1.info();
	c1.show();

	namedWindow("roi", CV_WINDOW_NORMAL);
	imshow("roi", c1.roi());
	waitKey(0);

	c1 = CpuHolo(lena, baboon);
	c1.info();
	c1.show();

	c1.expand(512, 768);
	c1.info();
	c1.show();

	c1.expand(1536, 768);
	c1.info();
	c1.show();

	c1 = CpuHolo(lena, 512, 512);
	c1.info();
	c1.show();

	c1 = CpuHolo(baboon, lena, 1536, 1536);
	c1.info();
	c1.show();

	imshow("roi", c1.roi());
	waitKey(0);
	*/
}

void quadPropagate() {
	CpuHolo h(holo);
	int sx = h.totalWidth();
	int sy = h.totalHeight();
	double z = 0.66;
	CpuHolo acc(sx * 2, sy * 2);

	h.expand(h.totalWidth() * 2, h.totalHeight() * 2);
	h.info();
	h.adjustROI(h.roiX(), h.roiY(), h.roiWidth() / 2, h.roiHeight() / 2);
	h.info();
	h.show(CpuHolo::dtRe);
	h.propagate(z);
	acc += h;

	h = CpuHolo(holo, sx * 2, sy * 2);
	h.info();
	h.adjustROI(h.roiX() * 1.5, h.roiY(), h.roiWidth() / 2, h.roiHeight() / 2);
	h.info();
	h.propagate(z);
	h.show(CpuHolo::dtRe);
	acc += h;

	h = CpuHolo(holo, sx * 2, sy * 2);
	h.adjustROI(h.roiX(), h.roiY()* 1.5, h.roiWidth() / 2, h.roiHeight() / 2);
	h.propagate(z);
	h.show(CpuHolo::dtRe);
	acc += h;

	h = CpuHolo(holo, sx * 2, sy * 2);
	h.adjustROI(h.roiX()* 1.5, h.roiY()* 1.5, h.roiWidth() / 2, h.roiHeight() / 2);
	h.propagate(z);
	h.show(CpuHolo::dtRe);
	acc += h;

	Mat intensity = acc.intensity();
	intensity.convertTo(intensity, CV_8UC1, 255.0);
	cv::equalizeHist(intensity, intensity);
	namedWindow("histeq", CV_WINDOW_NORMAL);
	imshow("histeq", intensity);
	waitKey(0);
}

int main(int argc, char* argv[])
{
	run_gpu_tests();

	return 0;
}

