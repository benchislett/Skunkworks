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
cv::Ptr<cv::BackgroundSubtractor> backsub;
double width;
double height;

cv::Mat frame;
cv::Mat fgmask;
cv::Mat aligned_frame;
cv::Mat cleaned_frame;
cv::Mat homography;

void init_cap(int video) {
  cap     = std::make_unique<cv::VideoCapture>(video);
  width   = (double) cap->get(cv::CAP_PROP_FRAME_WIDTH);
  height  = (double) cap->get(cv::CAP_PROP_FRAME_HEIGHT);
  backsub = cv::createBackgroundSubtractorMOG2();
}

cv::Mat& get_frame() {
  *cap >> frame;
  backsub->apply(frame, fgmask);
  return frame;
}

cv::Mat& get_aligned_frame() {
  get_frame();
  cv::warpPerspective(frame, aligned_frame, homography, frame.size());
  return aligned_frame;
}

cv::Mat& get_processed_frame() {
  get_frame();
  cleaned_frame.setTo(0);
  aligned_frame.setTo(0);
  cv::bitwise_and(frame, frame, aligned_frame, fgmask);
  cv::warpPerspective(aligned_frame, cleaned_frame, homography, cleaned_frame.size());
  return cleaned_frame;
}

int main(int argc, char** argv) {
  init_cap(0);

  std::vector<cv::Point2d> corner_points = {{0, 0}, {0, height}, {width, height}, {width, 0}};
  std::vector<cv::Point2d> homography_points;

  printf("Initializing background... Press SPACE to advance.\n");
  while (1) {
    get_frame();
    imshow(window, frame);
    if (cv::waitKey(1) == ' ')
      break;
  }
  /*
  while (background.empty()) {
    cap >> frame;
    imshow(window, frame);

    int key = cv::waitKey(1);
    if (key == ' ') {
      cv::FileStorage file("config.json", cv::FileStorage::FORMAT_JSON | cv::FileStorage::WRITE);
      cap >> background;
      file.write("background", background);
    } else if (key == 'g') {
      cv::FileStorage file("config.json", cv::FileStorage::FORMAT_JSON | cv::FileStorage::READ);
      file["background"] >> background;
      file["homography"] >> homography;
    }
  }*/

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
    // cv::FileStorage file("config.json", cv::FileStorage::FORMAT_JSON | cv::FileStorage::WRITE);
    homography = cv::findHomography(homography_points, corner_points);
    // file.write("homography", homography);
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

    imshow(window, tmp);
  }

  return 0;
}
