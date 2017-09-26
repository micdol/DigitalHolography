#include "CpuHolo.hpp"
#include <omp.h>

const double CpuHolo::CPUHOLO_DEFAULT_PITCH_X = 4.29e-6;
const double CpuHolo::CPUHOLO_DEFAULT_PITCH_Y = 4.29e-6;
const double CpuHolo::CPUHOLO_DEFAULT_LAMBDA = 632.8e-9;

CpuHolo::CpuHolo(const String &reFileName) :
	pitch_x(CPUHOLO_DEFAULT_PITCH_X),
	pitch_y(CPUHOLO_DEFAULT_PITCH_Y),
	lambda(CPUHOLO_DEFAULT_LAMBDA)
{
	load(reFileName);
}

CpuHolo::CpuHolo(const String & reFileName, int width, int height) :
	pitch_x(CPUHOLO_DEFAULT_PITCH_X),
	pitch_y(CPUHOLO_DEFAULT_PITCH_Y),
	lambda(CPUHOLO_DEFAULT_LAMBDA),
	_data(Mat(height, width, CV_64FC2)),
	_totalWidth(width),
	_totalHeight(height)
{
	loadCentered(reFileName);

}

CpuHolo::CpuHolo(const String &reFileName, const String &imFileName) :
	pitch_x(CPUHOLO_DEFAULT_PITCH_X),
	pitch_y(CPUHOLO_DEFAULT_PITCH_Y),
	lambda(CPUHOLO_DEFAULT_LAMBDA)
{
	load(reFileName, imFileName);
}

CpuHolo::CpuHolo(const String & reFileName, const String & imFileName, int width, int height) :
	pitch_x(CPUHOLO_DEFAULT_PITCH_X),
	pitch_y(CPUHOLO_DEFAULT_PITCH_Y),
	lambda(CPUHOLO_DEFAULT_LAMBDA),
	_data(Mat(height, width, CV_64FC2)),
	_totalWidth(width),
	_totalHeight(height)
{
	loadCentered(reFileName, imFileName);
}

CpuHolo::CpuHolo(int width, int height, const Scalar &val) :
	pitch_x(CPUHOLO_DEFAULT_PITCH_X),
	pitch_y(CPUHOLO_DEFAULT_PITCH_Y),
	lambda(CPUHOLO_DEFAULT_LAMBDA),
	_data(Mat(height, width, CV_64FC2, val)),
	_roiWidth(width),
	_roiHeight(height),
	_totalWidth(width),
	_totalHeight(height),
	_roiX(0),
	_roiY(0)
{}

CpuHolo::CpuHolo(int width, int height, double pitch_x_, double pitch_y_, double lambda_, const Scalar &val) :
	_data(Mat(height, width, CV_64FC2, val)),
	_roiX(0), _roiY(0),
	_roiWidth(width),
	_roiHeight(height),
	_totalWidth(width),
	_totalHeight(height),
	pitch_x(pitch_x_),
	pitch_y(pitch_y_),
	lambda(lambda_)
{}

CpuHolo::CpuHolo(int width, int height, uchar * data) :
	pitch_x(CPUHOLO_DEFAULT_PITCH_X),
	pitch_y(CPUHOLO_DEFAULT_PITCH_Y),
	lambda(CPUHOLO_DEFAULT_LAMBDA)
{
	// TODO: check this impl when cam is ready
	_data = Mat(height, width, CV_64FC2);
	_roiWidth = _data.cols;
	_roiHeight = _data.rows;
	_totalWidth = _roiWidth;
	_totalHeight = _roiHeight;
	_roiX = 0;
	_roiY = 0;
}

CpuHolo::CpuHolo(int width, int height, int data_width, int data_height, uchar * data) :
	pitch_x(CPUHOLO_DEFAULT_PITCH_X),
	pitch_y(CPUHOLO_DEFAULT_PITCH_Y),
	lambda(CPUHOLO_DEFAULT_LAMBDA)
{
	// TODO: check this impl when cam is ready
	_data = Mat(height, width, CV_64FC2);
	_roiWidth = _data.cols;
	_roiHeight = _data.rows;
	_totalWidth = _roiWidth;
	_totalHeight = _roiHeight;
	_roiX = 0;
	_roiY = 0;
}

CpuHolo::~CpuHolo()
{
	_data.release();
}

Mat CpuHolo::absSq(bool displayable) const {
	Mat retval(_data.size(), CV_64FC1, Scalar(0));
#pragma omp parallel for
	for (int i = 0; i < _data.rows; i++) {
		for (int j = 0; j < _data.cols; j++) {
			const Vec2d &p = _data.at<Vec2d>(i, j);
			retval.at<double>(i, j) = p[0] * p[0] + p[1] * p[1];
		}
	}
	if (displayable) {
		normalize(retval, retval, 0.0, 1.0, NORM_MINMAX);
	}
	return retval;
}

Mat CpuHolo::arg(bool displayable) const {
	Mat retval(_data.size(), CV_64FC1, Scalar(0));
#pragma omp parallel for
	for (int i = 0; i < _data.rows; i++) {
		for (int j = 0; j < _data.cols; j++) {
			const Vec2d &p = _data.at<Vec2d>(i, j);
			retval.at<double>(i, j) = atan2(p[1], p[0]);
		}
	}
	if (displayable) {
		normalize(retval, retval, 0.0, 1.0, NORM_MINMAX);
	}
	return retval;
}

Mat CpuHolo::im(bool displayable) const {
	vector<Mat> parts(2);
	split(_data, parts);
	Mat im = parts[1].clone();
	if (displayable) {
		normalize(im, im, 0.0, 1.0, NORM_MINMAX);
	}
	return im;
}

Mat CpuHolo::intensity(bool displayable) const {
	return absSq(displayable);
}

Mat CpuHolo::logAbs(bool displayable) const {
	Mat retval(_data.size(), CV_64FC1, Scalar(0));
	normalize(retval, retval, 0.0, 1.0, NORM_MINMAX);
#pragma omp parallel for
	for (int i = 0; i < _data.rows; i++) {
		for (int j = 0; j < _data.cols; j++) {
			const Vec2d &p = _data.at<Vec2d>(i, j);
			retval.at<double>(i, j) = log(1.0 + p[0] * p[0] + p[1] * p[1]);
		}
	}
	if (displayable) {
		normalize(retval, retval, 0.0, 1.0, NORM_MINMAX);
	}
	return retval;
}

Mat CpuHolo::re(bool displayable) const {
	vector<Mat> parts(2);
	split(_data, parts);
	Mat re = parts[0].clone();
	if (displayable) {
		normalize(re, re, 0.0, 1.0, NORM_MINMAX);
	}
	return re;
}

Mat CpuHolo::roi(bool displayable) const {
	// TODO roi should be 1 or 2 channel?
	Mat roi(absSq(), Rect(_roiX, _roiY, _roiWidth, _roiHeight));
	if (displayable) {
		normalize(roi, roi, 0.0, 1.0, NORM_MINMAX);
	}
	return roi;
}

void CpuHolo::adjustROI(int x, int y, int w, int h)
{
	Mat mask(_data.size(), CV_64FC1, Scalar(0.0));
	mask(Rect(x, y, w, h)) = Vec2d(1.0);
	vector<Mat> parts({ mask, mask });
	merge(parts, mask);

	multiply(_data, mask, _data);
	_roiX = x;
	_roiY = y;
	_roiWidth = w;
	_roiHeight = h;
}

void CpuHolo::expand(int tn_, const Scalar &val)
{
	expand(tn_, tn_, val);
}

void CpuHolo::expand(int tnx, int tny, const Scalar &val)
{
	// crop cols if less than current _data.cols
	if (tnx < _data.cols) {
		int start = static_cast<int>(_data.cols / 2.0 - tnx / 2.0);
		int end = static_cast<int>(_data.cols / 2.0 + tnx / 2.0);
		_roiX = 0;
		_roiWidth = tnx;
		_data(Range::all(), Range(start, end)).copyTo(_data);
	}
	// add cols if more then current _data.cols
	else {
		int col_cnt = static_cast<int>((tnx - _data.cols) / 2.0);
		_roiX = col_cnt;
		_roiWidth = _data.cols;
		copyMakeBorder(_data, _data, 0, 0, col_cnt, col_cnt, BORDER_CONSTANT, val);
	}
	// crop rows if more than current _data.rows
	if (tny < _data.rows) {
		int start = static_cast<int>(_data.rows / 2.0 - tny / 2.0);
		int end = static_cast<int>(_data.rows / 2.0 + tny / 2.0);
		_roiY = 0;
		_roiHeight = tny;
		_data(Range(start, end), Range::all()).copyTo(_data);
	}
	// expand rows if less then current _data.rows
	else {
		int row_cnt = static_cast<int>((tnx - _data.rows) / 2.0);
		_roiY = row_cnt;
		_roiHeight = _data.rows;
		copyMakeBorder(_data, _data, row_cnt, row_cnt, 0, 0, BORDER_CONSTANT, val);
	}
	_totalWidth = tnx;
	_totalHeight = tny;
}

void CpuHolo::fft() {
	dft(_data, _data, DFT_COMPLEX_OUTPUT & DFT_SCALE);
}

void CpuHolo::ifft() {
	idft(_data, _data, DFT_COMPLEX_OUTPUT & DFT_SCALE);
}

void CpuHolo::info()
{
	cout << "CpuHolo: " << endl
		<< "Pitch: [" << pitch_x << ", " << pitch_y << "]" << endl
		<< "Wavelength: " << lambda << endl
		<< "Dims: [" << _totalWidth << ", " << _totalHeight << "]" << endl
		<< "ROI: Start[x,y] = [" << _roiX << ", " << _roiY << "]\tDimensions[x,y]: [" << _roiWidth << ", " << _roiHeight << "]" << endl;
}

void CpuHolo::load(const String & reFileName)
{
	load(reFileName, "");
}

void CpuHolo::load(const String & reFileName, const String & imFileName)
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
	cv::merge(parts, 2, _data);

	_roiWidth = _data.cols;
	_roiHeight = _data.rows;
	_totalWidth = _roiWidth;
	_totalHeight = _roiHeight;
	_roiX = 0;
	_roiY = 0;
}

void CpuHolo::loadCentered(const String & reFileName, const Scalar & border)
{
	loadCentered(reFileName, "", border);
}

void CpuHolo::loadCentered(const String & reFileName, const String & imFileName, const Scalar & border)
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
	Mat re_im;
	cv::merge(parts, 2, re_im);

	// crop cols if more than current _data.cols
	if (re_im.cols > _data.cols) {
		int start = static_cast<int>((re_im.cols - _data.cols) / 2.0);
		_roiX = 0;
		_roiWidth = _data.cols;
		re_im = re_im(Range::all(), Range(start, start + _data.cols));
	}
	// expand cols if less then current _data.cols
	else {
		int col_cnt = static_cast<int>((_data.cols - re_im.cols) / 2.0);
		_roiX = col_cnt;
		_roiWidth = re_im.cols;
		copyMakeBorder(re_im, re_im, 0, 0, col_cnt, col_cnt, BORDER_CONSTANT, border);
	}
	// crop rows if more than current _data.rows
	if (re_im.rows > _data.rows) {
		int start = static_cast<int>((re_im.rows - _data.rows) / 2.0);
		_roiY = 0;
		_roiHeight = _data.rows;
		re_im = re_im(Range(start, start + _data.rows), Range::all());
	}
	// expand rows if less then current _data.rows
	else {
		int row_cnt = static_cast<int>((_data.rows - re_im.rows) / 2.0); 
		_roiY = row_cnt;
		_roiHeight = re_im.rows;
		copyMakeBorder(re_im, re_im, row_cnt, row_cnt, 0, 0, BORDER_CONSTANT, border);
	}

	_data = re_im;
}

void CpuHolo::mulTransferFunction(double z, double highPassFilterRadius) {
	struct ElementMul {
		double px, py, nx, ny, lambda, z, r;
		ElementMul(double px_, double py_, double nx_, double ny_, double lambda_, double z_, double r_) :
			px(px_), py(py_), nx(nx_), ny(ny_), lambda(lambda_), z(z_), r(r_) {};
		void operator ()(Vec2d &pixel, const int * pos) const {
			double dx = static_cast<double>(pos[0]) - nx / 2.0;
			double dy = static_cast<double>(pos[1]) - ny / 2.0;
			if (dx*dx + dy*dy < r) {
				pixel = 0.0;
			}
			else
			{
				double phase = (CV_2PI*z) / lambda - CV_PI*((dx *dx) / (nx*nx*px*px) + (dy * dy) / (ny*ny*py*py))*z*lambda;
				double a = pixel[0];
				double b = pixel[1];
				double c = cos(phase);
				double d = sin(phase);
				pixel = Vec2d(a*c - b*d, b*c + a*d);
			}
		}
	};
	_data.forEach<Vec2d>(ElementMul(pitch_x, pitch_y, _totalWidth, _totalHeight, lambda, z, highPassFilterRadius));
}

void CpuHolo::highPassFilter(double radius)
{
	struct ElementMul {
		double nx, ny, r;
		ElementMul(double nx_, double ny_, double r_) : nx(nx_), ny(ny_), r(r_) {};
		void operator ()(Vec2d &pixel, const int * pos) const {
			double dx = static_cast<double>(pos[0]) - nx / 2.0;
			double dy = static_cast<double>(pos[1]) - ny / 2.0;
			if (dx*dx + dy*dy < r) {
				pixel = 0.0;
			}
		}
	};
	_data.forEach<Vec2d>(ElementMul(_totalWidth, _totalHeight, radius));
}

void CpuHolo::propagate(double z) {
	cout << "Propagate for: " << z << endl;
	cout << "FFT...";
	swapQuadrants();
	fft();
	swapQuadrants();
	cout << " Done!" << endl;

	cout << "Mul imp resp...";
	mulTransferFunction(z);
	cout << " Done!" << endl;

	cout << "IFFT...";
	swapQuadrants();
	ifft();
	swapQuadrants();
	cout << " Done!" << endl;
}

void CpuHolo::swapQuadrants()
{
	int cx = _data.cols / 2;
	int cy = _data.rows / 2;

	Mat q0(_data, Rect(0, 0, cx, cy));
	Mat q1(_data, Rect(cx, 0, cx, cy));
	Mat q2(_data, Rect(0, cy, cx, cy));
	Mat q3(_data, Rect(cx, cy, cx, cy));

	Mat tmp;
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
}

void CpuHolo::save(const String &fileName, DataType saveWhat)
{
	Mat img;
	vector<Mat> parts(2);
	cv::split(_data, parts);
	switch (saveWhat) {
	case dtRe:
		img = parts[0].clone();
		normalize(img, img, 0.0, 1.0, NORM_MINMAX);
		img.convertTo(img, CV_16UC1, 255.0);
		imwrite(fileName, img);
		break;
	case dtIm:
		img = parts[1].clone();
		normalize(img, img, 0.0, 1.0, NORM_MINMAX);
		img.convertTo(img, CV_16UC1, 255.0);
		imwrite(fileName, img);
		break;
	case dtInt:
		img = intensity();
		img.convertTo(img, CV_16UC1, 255.0);
		imwrite(fileName, img);
		break;
	case dtArg:
		img = arg();
		img.convertTo(img, CV_16UC1, 255.0);
		imwrite(fileName, img);
		break;
	default: cout << "CpuHolo::save() - unknown option" << endl;
	}
}

void CpuHolo::show(int showWhat)
{
	vector<Mat> parts(2);
	cv::split(_data, parts);
	if (showWhat & dtRe) {
		Mat img = parts[0].clone();
		normalize(img, img, 0.0, 1.0, NORM_MINMAX);
		namedWindow("re", CV_WINDOW_NORMAL);
		imshow("re", img);
		resizeWindow("re", 512, 512);
	}
	if (showWhat & dtIm) {
		Mat img = parts[1].clone();
		normalize(img, img, 0.0, 1.0, NORM_MINMAX);
		namedWindow("im", CV_WINDOW_NORMAL);
		imshow("im", img);
		resizeWindow("im", 512, 512);
	}
	if (showWhat & dtInt) {
		Mat img = intensity();
		namedWindow("intensity", CV_WINDOW_NORMAL);
		imshow("intensity", img);
		resizeWindow("intensity", 512, 512);
	}
	if (showWhat & dtArg) {
		Mat img = arg();
		namedWindow("arg", CV_WINDOW_NORMAL);
		imshow("arg", img);
		resizeWindow("arg", 512, 512);
	}
	waitKey(0);
	destroyAllWindows();
}

const CpuHolo & CpuHolo::operator*=(const CpuHolo & other)
{
	for (int i = 0; i < _totalWidth; i++) {
		for (int j = 0; j < _totalHeight; j++) {
			Vec2d &lhs = _data.at<Vec2d>(i, j);
			const Vec2d &rhs = other._data.at<Vec2d>(i, j);
			double re = lhs[0] * rhs[0] - lhs[1] * rhs[1];
			double im = lhs[1] * rhs[0] + lhs[0] * rhs[1];
			lhs[0] = re;
			lhs[1] = im;
		}
	}
	return *this;
}

const CpuHolo & CpuHolo::operator+=(const CpuHolo & other)
{
	for (int i = 0; i < _totalWidth; i++) {
		for (int j = 0; j < _totalHeight; j++) {
			_data.at<Vec2d>(i, j) += other._data.at<Vec2d>(i, j);
		}
	}
	return *this;
}

