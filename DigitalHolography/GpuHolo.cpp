#include "GpuHolo.hpp"
#include "GpuHoloKernels.cuh"

const double GpuHolo::GPUHOLO_DEFAULT_PITCH_X = 4.29e-6;
const double GpuHolo::GPUHOLO_DEFAULT_PITCH_Y = 4.29e-6;
const double GpuHolo::GPUHOLO_DEFAULT_LAMBDA = 632.8e-9;

GpuHolo::GpuHolo(const String &reFileName) :
	pitch_x(GPUHOLO_DEFAULT_PITCH_X),
	pitch_y(GPUHOLO_DEFAULT_PITCH_Y),
	lambda(GPUHOLO_DEFAULT_LAMBDA)
{
	load(reFileName);
}

GpuHolo::GpuHolo(const String & reFileName, int width, int height) :
	pitch_x(GPUHOLO_DEFAULT_PITCH_X),
	pitch_y(GPUHOLO_DEFAULT_PITCH_Y),
	lambda(GPUHOLO_DEFAULT_LAMBDA), 
	_data(Mat(height, width, CV_64FC2)),
	_totalWidth(width),
	_totalHeight(height)
{
	loadCentered(reFileName);
}

GpuHolo::GpuHolo(const String &reFileName, const String &imFileName) :
	pitch_x(GPUHOLO_DEFAULT_PITCH_X),
	pitch_y(GPUHOLO_DEFAULT_PITCH_Y),
	lambda(GPUHOLO_DEFAULT_LAMBDA)
{
	load(reFileName, imFileName);
}

GpuHolo::GpuHolo(const String & reFileName, const String & imFileName, int width, int height) : 
	pitch_x(GPUHOLO_DEFAULT_PITCH_X),
	pitch_y(GPUHOLO_DEFAULT_PITCH_Y),
	lambda(GPUHOLO_DEFAULT_LAMBDA),
	_data(Mat(height, width, CV_64FC2)),
	_totalWidth(width),
	_totalHeight(height)
{
	loadCentered(reFileName, imFileName);
}

GpuHolo::GpuHolo(int width, int height, const Scalar &val) :
	pitch_x(GPUHOLO_DEFAULT_PITCH_X),
	pitch_y(GPUHOLO_DEFAULT_PITCH_Y),
	lambda(GPUHOLO_DEFAULT_LAMBDA),
	_data(GpuMat(height, width, CV_64FC2, val)),
	_roiWidth(width),
	_roiHeight(height),
	_totalWidth(width),
	_totalHeight(height),
	_roiX(0),
	_roiY(0)
{}

GpuHolo::GpuHolo(int width, int height, double pitch_x_, double pitch_y_, double lambda_, const Scalar & val) :
	_data(GpuMat(height, width, CV_64FC2, val)),
	_roiX(0), _roiY(0),
	_roiWidth(width),
	_roiHeight(height),
	_totalWidth(width),
	_totalHeight(height),
	pitch_x(pitch_x_),
	pitch_y(pitch_y_),
	lambda(lambda_)
{}

GpuHolo::GpuHolo(int width, int height, uchar * data) :
	pitch_x(GPUHOLO_DEFAULT_PITCH_X),
	pitch_y(GPUHOLO_DEFAULT_PITCH_Y),
	lambda(GPUHOLO_DEFAULT_LAMBDA)
{
	// TODO GpuMat data constructor
}

GpuHolo::GpuHolo(int width, int height, int data_width, int data_height, uchar * data) :
	pitch_x(GPUHOLO_DEFAULT_PITCH_X),
	pitch_y(GPUHOLO_DEFAULT_PITCH_Y),
	lambda(GPUHOLO_DEFAULT_LAMBDA)
{
	// TODO GpuMat data centered constructor
}

GpuHolo::~GpuHolo()
{
	_data.release();
}

/*Mat GpuHolo::absSq(bool displayable)
{
	/*GpuMat output(_data.rows, _data.cols, CV_64FC1, Scalar(0));
	std::vector<GpuMat> parts(2);
	cv::cuda::split(_data, parts);
	cout << parts[0].type() << " " << parts[1].type() << endl;
	gpuholo::callAbsSqKernel(parts[0], parts[1], output);
	
	Mat retval;
	output.download(retval);
	cv::normalize(retval, retval, 0.0, 1.0, NORM_MINMAX);
	return retval;
}
*/

Mat GpuHolo::absSq(bool displayable) const
{
	Mat retval;
	absSqGpu(displayable).download(retval);
	return retval;
}

Mat GpuHolo::arg(bool displayable) const
{
	Mat retval;
	argGpu(displayable).download(retval);
	return retval;
}

Mat GpuHolo::im(bool displayable) const
{
	Mat retval;
	imGpu(displayable).download(retval);
	return retval;
}

Mat GpuHolo::intensity(bool displayable) const
{
	Mat retval;
	intensityGpu(displayable).download(retval);
	return retval;
}

Mat GpuHolo::logAbs(bool displayable) const
{
	Mat retval;
	logAbsGpu(displayable).download(retval);
	return retval;
}

Mat GpuHolo::re(bool displayable /*= true*/) const
{
	Mat retval;
	reGpu(displayable).download(retval);
	return retval;
}

Mat GpuHolo::roi(bool displayable /*= true*/) const
{
	Mat retval;
	roiGpu(displayable).download(retval);
	return retval;
}

GpuMat GpuHolo::absSqGpu(bool displayable) const
{
	GpuMat retval(_data.rows, _data.cols, CV_64FC1, Scalar(0));
	vector<GpuMat> parts(2);;
	cuda::split(_data, parts);
	gpu_holo::callAbsSqKernel(parts[0], parts[1], retval);
	if (displayable) {
		cuda::normalize(retval, retval, 0.0, 1.0, NORM_MINMAX, -1);
	}
	return retval;
}

GpuMat GpuHolo::argGpu(bool displayable) const
{
	GpuMat retval(_data.rows, _data.cols, CV_64FC1, Scalar(0));
	vector<GpuMat> parts(2);;
	cuda::split(_data, parts);
	gpu_holo::callArgKernel(parts[0], parts[1], retval);
	if (displayable) {
		cuda::normalize(retval, retval, 0.0, 1.0, NORM_MINMAX, -1);
	}
	return retval;
}

GpuMat GpuHolo::imGpu(bool displayable) const
{
	vector<GpuMat> parts(2);;
	cuda::split(_data, parts);
	GpuMat retval = parts[1].clone();
	if (displayable) {
		cuda::normalize(retval, retval, 0.0, 1.0, NORM_MINMAX, -1);
	}
	return retval;
}

GpuMat GpuHolo::intensityGpu(bool displayable) const
{
	return absSqGpu(displayable);
}

GpuMat GpuHolo::logAbsGpu(bool displayable) const
{
	GpuMat retval(_data.rows, _data.cols, CV_64FC1, Scalar(0));
	vector<GpuMat> parts(2);;
	cuda::split(_data, parts);
	gpu_holo::calllogAbsKernel(parts[0], parts[1], retval);
	if (displayable) {
		cuda::normalize(retval, retval, 0.0, 1.0, NORM_MINMAX, -1);
	}
	return retval;
}

GpuMat GpuHolo::reGpu(bool displayable) const
{
	vector<GpuMat> parts(2);;
	cuda::split(_data, parts);
	GpuMat retval = parts[0].clone();
	if (displayable) {
		cuda::normalize(retval, retval, 0.0, 1.0, NORM_MINMAX, -1);
	}
	return retval;
}

GpuMat GpuHolo::roiGpu(bool displayable) const
{
	// TODO roi should be 1 or 2 channel
	GpuMat roi(absSqGpu(), Rect(_roiX, _roiY, _roiWidth, _roiHeight));
	if (displayable) {
		cuda::normalize(roi, roi, 0.0, 1.0, NORM_MINMAX, -1);
	}
	return roi;
}

void GpuHolo::adjustROI(int x, int y, int w, int h)
{
}

void GpuHolo::expand(int tn_, const Scalar & val)
{
}

void GpuHolo::expand(int tnx, int tny, const Scalar & val)
{
}

void GpuHolo::fft()
{
}

void GpuHolo::ifft()
{
}

void GpuHolo::info()
{
}
 
void GpuHolo::load(const String & reFileName)
{
	load(reFileName, "");
}

void GpuHolo::load(const String & reFileName, const String & imFileName)
{
	Mat re = imread(reFileName, CV_LOAD_IMAGE_GRAYSCALE);
	re.convertTo(re, CV_64FC1, 1.0 / 255.0);
	
	cout << re.size() << endl;

	Mat im = imread(imFileName, CV_LOAD_IMAGE_GRAYSCALE);
	if (!im.data) {
		im = Mat(re.size(), CV_64FC1, Scalar(0));
	}
	else {
		im.convertTo(im, CV_64FC1, 1.0 / 255.0);
	}
	cout << im.size() << endl;

	Mat parts[] = { re, im };
	Mat merged;
	cv::merge(parts, 2, merged);
	cout << merged.size() << endl;

	_data.upload(merged);
	cout << _data.size() << endl;

	_roiWidth = _data.cols;
	_roiHeight = _data.rows;
	_totalWidth = _roiWidth;
	_totalHeight = _roiHeight;
	_roiX = 0;
	_roiY = 0;
}

void GpuHolo::loadCentered(const String & reFileName, const Scalar & border)
{
	loadCentered(reFileName, "", border);
}

void GpuHolo::loadCentered(const String & reFileName, const String & imFileName, const Scalar & border)
{
	Mat re = imread(reFileName, CV_LOAD_IMAGE_GRAYSCALE);
	re.convertTo(re, CV_64FC1, 1.0 / 255.0);

	Mat im = imread(imFileName, CV_LOAD_IMAGE_GRAYSCALE);
	if (!im.data) {
		im = Mat(re.size(), CV_64FC1, Scalar(0));
	}
	else {
		im.convertTo(im, CV_64FC1, 1.0 / 255.0);
	}

	Mat parts[] = { re, im };
	Mat merged;
	cv::merge(parts, 2, merged);
	_data.upload(merged);

	// crop cols if more than current _data.cols
	if (_data.cols > _totalWidth ) {
		int start = static_cast<int>((_data.cols - _totalWidth) / 2.0);
		_roiX = 0;
		_roiWidth = _totalWidth;
		_data = _data(Range::all(), Range(start, start + _totalWidth));
	}
	// expand cols if less then current _data.cols
	else {
		int col_cnt = static_cast<int>((_totalWidth - _data.cols) / 2.0);
		_roiX = col_cnt;
		_roiWidth = _data.cols;
		cv::cuda::copyMakeBorder(_data, _data, 0, 0, col_cnt, col_cnt, BORDER_CONSTANT, border);
	}
	// crop rows if more than current _data.rows
	if (_data.rows > _totalHeight) {
		int start = static_cast<int>((_data.rows - _totalHeight) / 2.0);
		_roiY = 0;
		_roiHeight = _totalHeight;
		_data = _data(Range(start, start + _totalHeight), Range::all());
	}
	// expand rows if less then current _data.rows
	else {
		int row_cnt = static_cast<int>((_totalHeight - _data.rows) / 2.0);
		_roiY = row_cnt;
		_roiHeight = _data.rows;
		cv::cuda::copyMakeBorder(_data, _data, row_cnt, row_cnt, 0, 0, BORDER_CONSTANT, border);
	}
}

void GpuHolo::mulTransferFunction(double z, double highPassFilterRadius)
{
}

void GpuHolo::highPassFilter(double radius)
{
}

void GpuHolo::propagate(double z)
{
}

void GpuHolo::swapQuadrants()
{
	int cx = _data.cols / 2;
	int cy = _data.rows / 2;

	GpuMat q0(_data, Rect(0, 0, cx, cy));
	GpuMat q1(_data, Rect(cx, 0, cx, cy));
	GpuMat q2(_data, Rect(0, cy, cx, cy));
	GpuMat q3(_data, Rect(cx, cy, cx, cy));

	GpuMat tmp;
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
}

void GpuHolo::save(const String & fileName, DataType saveWhat)
{
}

void GpuHolo::show(int showWhat)
{
	GpuMat img;
	vector<GpuMat> parts(2);
	split(_data, parts);
	if (showWhat & dtRe) {
		img = parts[0].clone();
		cuda::normalize(img, img, 0.0, 1.0, NORM_MINMAX, -1);
		namedWindow("re", CV_WINDOW_NORMAL);
		imshow("re", img);
		resizeWindow("re", 512, 512);
	}
	if (showWhat & dtIm) {
		img = parts[1].clone();
		cuda::normalize(img, img, 0.0, 1.0, NORM_MINMAX, -1);
		namedWindow("im", CV_WINDOW_NORMAL);
		imshow("im", img);
		resizeWindow("im", 512, 512);
	}
	if (showWhat & dtInt) {
		img = intensityGpu();
		namedWindow("intensity", CV_WINDOW_OPENGL | CV_WINDOW_NORMAL);
		imshow("intensity", img);
		resizeWindow("intensity", 512, 512);
	}
	if (showWhat & dtArg) {
		img = argGpu();
		namedWindow("arg", CV_WINDOW_OPENGL | CV_WINDOW_NORMAL);
		imshow("arg", img);
		resizeWindow("arg", 512, 512);
	}
	waitKey(0);
	destroyAllWindows();
}
