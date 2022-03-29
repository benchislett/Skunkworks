#include <cstring>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/persistence.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>

const std::string window = "RoboSoccer";

int main(int argc, char** argv) {
  std::string video = "/dev/video2";
  cv::VideoCapture cap(video);
  cv::Mat frame;

  double width  = (double) cap.get(cv::CAP_PROP_FRAME_WIDTH);
  double height = (double) cap.get(cv::CAP_PROP_FRAME_HEIGHT);

  cv::Mat background;
  cv::Mat homography;

  std::vector<cv::Point2d> corner_points = {{0, 0}, {0, height}, {width, height}, {width, 0}};
  std::vector<cv::Point2d> homography_points;

  printf("Press SPACE to capture the background. Press 'g' to use precomputed data.\n");
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
  }

  cv::Rect roi;
  while (homography.empty() && homography_points.size() < 4) {
    cap >> frame;
    for (auto p : homography_points)
      cv::circle(frame, p, 8, {0, 255, 0}, cv::FILLED);
    imshow(window, frame);

    roi = cv::selectROI(window, frame);

    homography_points.emplace_back(roi.x + (roi.width / 2.0), roi.y + (roi.height / 2.0));
    printf("Corner identified at: (%f, %f)\n", homography_points.back().x, homography_points.back().y);
  }

  if (homography.empty()) {
    cv::FileStorage file("config.json", cv::FileStorage::FORMAT_JSON | cv::FileStorage::APPEND);
    homography = cv::findHomography(homography_points, corner_points);
    file.write("homography", homography);
  }

  cv::Mat transformed_frame;
  printf("Displaying homography. Press SPACE to continue.\n");
  while (cv::waitKey(1) != ' ') {
    cap >> frame;
    cv::warpPerspective(frame, transformed_frame, homography, frame.size());
    imshow(window, transformed_frame);
  }

  printf("Displaying background-subtracted homography. Press SPACE to continue.\n");
  while (cv::waitKey(1) != ' ') {
    cap >> frame;
    frame -= background;
    cv::warpPerspective(frame, transformed_frame, homography, frame.size());

    imshow(window, transformed_frame);
  }

  cv::Ptr<cv::Tracker> self_tracker = cv::TrackerCSRT::create();
  cv::Ptr<cv::Tracker> opp_tracker  = cv::TrackerCSRT::create();
  cv::Ptr<cv::Tracker> ball_tracker = cv::TrackerCSRT::create();

  printf("Identify self: ");
  cap >> frame;
  frame -= background;
  cv::warpPerspective(frame, transformed_frame, homography, frame.size());
  roi = selectROI(window, transformed_frame);
  self_tracker->init(transformed_frame, roi);
  printf("done!\n");

  printf("Identify opponent: ");
  cap >> frame;
  frame -= background;
  cv::warpPerspective(frame, transformed_frame, homography, frame.size());
  roi = selectROI(window, transformed_frame);
  opp_tracker->init(transformed_frame, roi);
  printf("done!\n");

  printf("Identify ball: ");
  cap >> frame;
  frame -= background;
  cv::warpPerspective(frame, transformed_frame, homography, frame.size());
  roi = selectROI(window, transformed_frame);
  ball_tracker->init(transformed_frame, roi);
  printf("done!\n");

  printf("Start the tracking process, press ESC to quit.\n");
  while (cv::waitKey(1) != 27) {
    cap >> frame;
    frame -= background;
    cv::warpPerspective(frame, transformed_frame, homography, frame.size());
    transformed_frame.copyTo(frame);
    self_tracker->update(transformed_frame, roi);
    rectangle(frame, roi, {255, 0, 0}, 2, 1);
    opp_tracker->update(transformed_frame, roi);
    rectangle(frame, roi, {0, 255, 0}, 2, 1);
    ball_tracker->update(transformed_frame, roi);
    rectangle(frame, roi, {0, 0, 255}, 2, 1);

    imshow(window, frame);
  }

  return 0;
}
