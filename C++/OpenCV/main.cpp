
#include <chrono>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/cvconfig.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <stdio.h>

using Clock = std::chrono::steady_clock;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::time_point;
using namespace std::literals::chrono_literals;

using namespace cv;
using namespace std;

void thresh_callback(int, void*);

RNG rng(12345);

Mat img, gray, edges, output;

int low  = 0;
int high = 255;

int main(int, char**) {
  img = imread("../im.jpg");
  cvtColor(img, gray, COLOR_BGR2GRAY);
  blur(gray, gray, Size(3, 3));

  namedWindow("opencv");
  createTrackbar("Canny low:", "opencv", &low, 512, thresh_callback);
  createTrackbar("Canny high:", "opencv", &high, 512, thresh_callback);
  thresh_callback(0, 0);

  while (waitKey() != 'q')
    ;
}


void thresh_callback(int, void*) {
  Canny(gray, edges, low, high);

  vector<vector<Point>> contours;
  vector<Vec4i> hierarchy;
  findContours(edges, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

  Mat output = Mat::zeros(edges.size(), CV_8UC3);
  for (size_t i = 0; i < contours.size(); i++) {
    Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
    drawContours(output, contours, (int) i, color, 2, LINE_8, hierarchy, 0);
  }

  // imshow("opencv", img);
  // while (pollKey() != 'n')
  //   ;
  // imshow("opencv", gray);
  // while (pollKey() != 'n')
  //   ;
  imshow("opencv", edges);
  // while (pollKey() != 'n')
  //   ;
  // imshow("opencv", output);
  // while (pollKey() != 'n')
  //   ;
}