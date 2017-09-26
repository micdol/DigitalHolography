#pragma once

#include <opencv2\opencv.hpp>

using namespace cv;
using namespace std;

class CpuHolo
{

private:
	/*
	  [0,0]			_tnx
		+---------------------------+
		|							|
		|	 [_x, _y]				|
		|		+-----------+		|
		|		|			|		|
		|		|	 ROI	| _ny	|	_tny
		|		|			|		|
		|		+-----------+		|
		|			 _nx			|
		|							|
		+---------------------------+
	*/

	// dimensions of the ROI (without zero padding if any)
	int _roiWidth, _roiHeight;
	// total dimensions of computational kernel, tn_ >= n_
	int _totalWidth, _totalHeight;
	// coords of top left corner of ROI inside computational kernel
	int _roiX, _roiY;
	// 2-channel data (re, im) 
	Mat _data;

public:
	static const double CPUHOLO_DEFAULT_PITCH_X;
	static const double CPUHOLO_DEFAULT_PITCH_Y;
	static const double CPUHOLO_DEFAULT_LAMBDA;

	enum DataType : char {
		dtRe = 0x01, 
		dtIm = 0x02,
		dtInt = 0x04,
		dtArg = 0x08
	};

	// pitch x, y and wavelength
	double pitch_x, pitch_y, lambda;

public:
	//constructors
	CpuHolo(const String &reFileName);
	CpuHolo(const String &reFileName, int width, int height);
	CpuHolo(const String &reFileName, const String &imFileName);
	CpuHolo(const String &reFileName, const String &imFileName, int width, int height);
	CpuHolo(int width, int height, const Scalar &val = Scalar(0));
	CpuHolo(int width, int height, double pitch_x, double pitch_y, double lambda, const Scalar &val = Scalar(0));
	CpuHolo(int width, int height, uchar *data);
	CpuHolo(int width, int height, int data_width, int data_height, uchar *data);
	~CpuHolo();

	// non-modifiable (externally) private members accessors
	int roiX() const { return _roiX; }
	int roiY() const { return _roiY; }
	int roiWidth() const { return _roiWidth; }
	int roiHeight() const { return _roiHeight; }
	int totalWidth() const { return _totalWidth; }
	int totalHeight() const { return _totalHeight; }

	Mat absSq(bool displayable = true) const;
	Mat arg(bool displayable = true) const;
	Mat im(bool displayable = true) const;
	Mat intensity(bool displayable = true) const;
	Mat logAbs(bool displayable = true) const;
	Mat re(bool displayable = true) const;
	Mat roi(bool displayable = true) const;
	 
	void adjustROI(int x, int y, int w, int h);
	void expand(int tn_, const Scalar &val = Scalar(0));
	void expand(int tnx, int tny, const Scalar &val = Scalar(0));
	void fft();
	void ifft();
	void info();
	void load(const String &reFileName);
	void load(const String &reFileName, const String &imFileName);
	void loadCentered(const String &reFileName, const Scalar & border = Scalar(0));
	void loadCentered(const String &reFileName, const String &imFileName, const Scalar & border = Scalar(0));
	void mulTransferFunction(double z, double highPassFilterRadius = 256.0);
	void highPassFilter(double radius);
	void propagate(double z);
	void swapQuadrants();
	void save(const String &fileName, DataType saveWhat = dtInt);
	void show(int showWhat = dtRe | dtIm);

	const CpuHolo& operator*=(const CpuHolo& other);
	const CpuHolo& operator+=(const CpuHolo& other);

	//TODO implement if needed
	const CpuHolo& operator*=(const Mat& other);
	const CpuHolo& operator*=(const Vec2d& cplx);
	const CpuHolo& operator*=(double scale);
	const CpuHolo& operator/=(const CpuHolo& other);
	const CpuHolo& operator/=(const Mat& other);
	const CpuHolo& operator/=(const Vec2d& cplx);
	const CpuHolo& operator/=(double scale);
	const CpuHolo& operator+=(const Mat& other);
	const CpuHolo& operator+=(const Vec2d& cplx);
	const CpuHolo& operator+=(double val);
	const CpuHolo& operator-=(const CpuHolo& other);
	const CpuHolo& operator-=(const Mat& other);
	const CpuHolo& operator-=(const Vec2d& cplx);
	const CpuHolo& operator-=(double val);
};
