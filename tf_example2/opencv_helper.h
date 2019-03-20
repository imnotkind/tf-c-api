#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/cvdef.h>

#include <iostream>
#include <vector>
#include <set>
#include <string>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstddef>
#include <cstdint>
#include <Windows.h>

using namespace std;
using namespace cv;


class Pallete {

public:
	vector<vector<int>> color_pallete;

	Pallete() {
		//opencv : BGR!!!
		color_pallete = {
			{ 0, 0, 0 },
			{0, 0, 255},
			{255, 0, 0},
			{0, 255, 0},
		};
		
	}
};

inline string type2str(int type) {
	string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	return r;
}



inline void showimage_fromMat(Mat image)
{
	cout << type2str(image.type()) << endl;
	cout << image.size() << endl;
	if (image.empty())                      // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return;
	}
	if (image.type() != CV_8UC1 && image.type() != CV_8UC3 && image.type() != CV_8UC4)
	{
		cout << "Not an 8bit unsigned channel 1 or 3 or 4 matrix" << std::endl;
		return;
	}
	namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("Display window", image);                // Show our image inside it.
	waitKey(0); // Wait for a keystroke in the window
}


//https://stackoverflow.com/questions/35993895/create-a-rgb-image-from-pixel-labels
inline void show_label_image(Mat label)
{
	if (label.empty())
	{
		cout << "pred Mat empty" << std::endl;
		return;
	}
	if (label.type() != CV_32SC1 && label.type() != CV_8UC1)
	{
		cout << "Not an 32bit signed or 8bit unsigned channel 1 matrix : " <<type2str(label.type()) << std::endl;
		return;
	}



	Pallete p;

	Mat pred2;

	if (label.type() == CV_32SC1)
		label.convertTo(pred2, CV_8UC1);
	else
		pred2 = label;


	cv::Mat draw;

	std::vector<cv::Mat> matChannels;
	cv::split(pred2, matChannels);
	matChannels.push_back(pred2);
	matChannels.push_back(pred2);
	cv::merge(matChannels, draw);


	draw.forEach<Vec3b>
	(
		[p](Vec3b &pixel, const int * position) -> void
		{
			vector<int> t;
			if (p.color_pallete.size() > pixel[0])
				t = p.color_pallete[pixel[0]];
			else
			{
				cout << "out of pallete range" << endl;
				t = p.color_pallete[0];
			}
				
			pixel[0] = t[0];
			pixel[1] = t[1];
			pixel[2] = t[2];
		}
	);


	showimage_fromMat(draw);


}