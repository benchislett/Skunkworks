#include <cstring>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/persistence.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/videoio.hpp>

const std::string window = "RoboSoccer";

std::unique_ptr<cv::VideoCapture> cap = nullptr;
double width;
double height;

cv::Mat frame;
cv::Mat fgmask;
cv::Mat background;
cv::Mat aligned_frame;
cv::Mat cleaned_frame;
cv::Mat homography;

void init_cap(int video) {
  cap    = std::make_unique<cv::VideoCapture>(video);
  width  = (double) cap->get(cv::CAP_PROP_FRAME_WIDTH);
  height = (double) cap->get(cv::CAP_PROP_FRAME_HEIGHT);
}

cv::Mat& get_frame() {
  *cap >> frame;
  return frame;
}

cv::Mat& get_aligned_frame() {
  get_frame();
  cv::warpPerspective(frame, aligned_frame, homography, frame.size());
  return aligned_frame;
}

cv::Mat& get_processed_frame() {
  get_frame();

  // aligned_frame.setTo(0);
  cv::absdiff(frame, background, fgmask);
  cv::cvtColor(fgmask, fgmask, cv::COLOR_BGR2GRAY);
  // fgmask = fgmask > 1;
  // cv::bitwise_and(frame, frame, aligned_frame, fgmask);
  frame.copyTo(aligned_frame);
  cv::warpPerspective(aligned_frame, cleaned_frame, homography, aligned_frame.size());
  return cleaned_frame;
}

int main(int argc, char** argv) {
  init_cap(0);

  std::vector<cv::Point2d> corner_points = {{0, 0}, {0, height}, {width, height}, {width, 0}};
  std::vector<cv::Point2d> homography_points;

  printf("Initializing background... Press SPACE to capture.\n");
  while (background.empty()) {
    imshow(window, get_frame());

    int key = cv::waitKey(1);
    if (key == ' ') {
      frame.copyTo(background);
    }
  }

  printf("Select field corners for homography.\n");
  cv::Rect roi;
  while (homography.empty() && homography_points.size() < 4) {
    get_frame();
    for (auto p : homography_points)
      cv::circle(frame, p, 8, {0, 255, 0}, cv::FILLED);
    imshow(window, frame);

    roi = cv::selectROI(window, frame);

    homography_points.emplace_back(roi.x + (roi.width / 2.0), roi.y + (roi.height / 2.0));
    printf("Corner identified at: (%f, %f)\n", homography_points.back().x, homography_points.back().y);
  }

  if (homography.empty()) {
    homography = cv::findHomography(homography_points, corner_points);
  }

  printf("Displaying homography. Press SPACE to continue.\n");
  while (cv::waitKey(1) != ' ') {
    imshow(window, get_aligned_frame());
  }

  printf("Displaying background-subtracted homography. Press SPACE to continue.\n");
  while (cv::waitKey(1) != ' ') {
    imshow(window, get_processed_frame());
    imshow("mask", fgmask);
  }

  cv::Ptr<cv::Tracker> self_tracker = cv::TrackerCSRT::create();
  cv::Ptr<cv::Tracker> opp_tracker  = cv::TrackerCSRT::create();
  cv::Ptr<cv::Tracker> ball_tracker = cv::TrackerCSRT::create();

  printf("Identify self: ");
  roi = selectROI(window, get_processed_frame());
  self_tracker->init(cleaned_frame, roi);
  printf("done!\n");

  printf("Identify opponent: ");
  roi = selectROI(window, get_processed_frame());
  opp_tracker->init(cleaned_frame, roi);
  printf("done!\n");

  printf("Identify ball: ");
  roi = selectROI(window, get_processed_frame());
  ball_tracker->init(cleaned_frame, roi);
  printf("done!\n");

  printf("Start the tracking process, press ESC to quit.\n");
  cv::Mat tmp;
  while (cv::waitKey(1) != 27) {
    get_processed_frame().copyTo(tmp);
    self_tracker->update(cleaned_frame, roi);
    rectangle(tmp, roi, {255, 0, 0}, 2, 1);
    opp_tracker->update(cleaned_frame, roi);
    rectangle(tmp, roi, {0, 255, 0}, 2, 1);
    ball_tracker->update(cleaned_frame, roi);
    rectangle(tmp, roi, {0, 0, 255}, 2, 1);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::Mat gray_cleaned_frame;
    cv::cvtColor(cleaned_frame, gray_cleaned_frame, cv::COLOR_BGR2GRAY);
    cv::Mat canny;
    cv::Canny(gray_cleaned_frame, canny, 50, 100, 3);
    imshow("Canny", canny);
    cv::findContours(canny, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Point> approxTriangle;
    for (size_t i = 0; i < contours.size(); i++) {
      cv::approxPolyDP(contours[i], approxTriangle, cv::arcLength(cv::Mat(contours[i]), true) * 0.05, true);
      if (approxTriangle.size() == 3) {
        std::vector<cv::Point>::iterator vertex;
        for (vertex = approxTriangle.begin(); vertex != approxTriangle.end(); ++vertex) {
          cv::circle(tmp, *vertex, 3, {0, 255, 0}, cv::FILLED);
        }
      }
    }

    imshow(window, tmp);
  }

  return 0;
}
