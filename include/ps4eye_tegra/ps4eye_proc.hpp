#ifndef PS4EYE_TEGRA_PS4EYE_PROC_HPP_
#define PS4EYE_TEGRA_PS4EYE_PROC_HPP_

#include <boost/version.hpp>
#if ((BOOST_VERSION / 100) % 1000) >= 53
#include <boost/thread/lock_guard.hpp>
#endif

#include <ros/ros.h>
#include <ros/package.h>
#include <nodelet/nodelet.h>

#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <image_geometry/stereo_camera_model.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <cv_bridge/cv_bridge.h>

// CUDA
#if OPENCV3
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/cudawarping.hpp>
#define GPUNS cv::cuda
#else
#include <opencv2/gpu/gpu.hpp>
#define GPUNS cv::gpu
#endif
//

#include <sensor_msgs/image_encodings.h>
#include <stereo_msgs/DisparityImage.h>

#include <yaml-cpp/yaml.h>

using namespace sensor_msgs;
using namespace stereo_msgs;
using namespace message_filters::sync_policies;

namespace ps4eye_tegra {

class PS4EyeProc : public nodelet::Nodelet {
  typedef ExactTime<Image, CameraInfo> ExactPolicy;
  typedef ApproximateTime<Image, CameraInfo> ApproximatePolicy;
  typedef message_filters::Synchronizer<ExactPolicy> ExactSync;
  typedef message_filters::Synchronizer<ApproximatePolicy> ApproximateSync;

 public:
  virtual void onInit();
  void connectCallback();
  void imageCallback(const ImageConstPtr& image_msg,
                     const CameraInfoConstPtr& info_msg);
 private:
  boost::shared_ptr<image_transport::ImageTransport> it_;
  boost::shared_ptr<image_transport::ImageTransport> it_left_camera_;

  image_transport::SubscriberFilter sub_image_;
  message_filters::Subscriber<CameraInfo> sub_info_;

  boost::mutex connection_mutex_;
  ros::Publisher pub_disparity_;
  ros::Publisher pub_right_info_;
  image_transport::CameraPublisher pub_left_camera_;

  boost::shared_ptr<ExactSync> exact_sync_;
  boost::shared_ptr<ApproximateSync> approximate_sync_;

  // camera image
  cv_bridge::CvImage left_cvimage_;
  GPUNS::GpuMat gpu_input_;

  // camera_info
  sensor_msgs::CameraInfo left_info_;
  sensor_msgs::CameraInfo right_info_;

  // crop
  uint32_t l_x_offset_;
  uint32_t l_y_offset_;
  uint32_t l_width_;
  uint32_t l_height_;
  uint32_t r_x_offset_;
  uint32_t r_y_offset_;
  uint32_t r_width_;
  uint32_t r_height_;

  // rectify
  image_geometry::PinholeCameraModel left_model_;
  image_geometry::PinholeCameraModel right_model_;
  cv::Mat left_map1_, left_map2_;
  GPUNS::GpuMat gpu_left_map1_, gpu_left_map2_;
  GPUNS::GpuMat gpu_left_rect_color_, gpu_left_rect_;
  cv::Mat right_map1_, right_map2_;
  GPUNS::GpuMat gpu_right_map1_, gpu_right_map2_;
  GPUNS::GpuMat gpu_right_rect_color_, gpu_right_rect_;

  // stretch
  int32_t stretch_factor_;
  GPUNS::GpuMat gpu_left_stretch_;
  GPUNS::GpuMat gpu_right_stretch_;

#if OPENCV3
  cv::cuda::HostMem left_rect_color_;
#else
  cv::Mat left_rect_color_;
#endif

  // stereo matching
  image_geometry::StereoCameraModel stereo_model_;
#if OPENCV3
  mutable cv::Ptr<cv::cuda::StereoBM> block_matcher_;
  mutable cv::Ptr<cv::cuda::StereoConstantSpaceBP> csbp_matcher_;
  mutable cv::Ptr<cv::cuda::DisparityBilateralFilter> bilateral_filter_;
  mutable cv::cuda::HostMem disparity_;
#else
  mutable cv::gpu::StereoBM_GPU block_matcher_;
  mutable cv::Mat disparity_;
#endif
  int win_size_;
  int ndisp_;
  int filter_radius_;
  int filter_iter_;
  GPUNS::GpuMat gpu_disp_, gpu_disp_filtered_, gpu_disp_stretch_;

  // switch
  bool use_csbp_;
  bool use_bilateral_filter_;
  bool use_stretch_;

  void readCameraInfo_(const std::string& filename,
                       sensor_msgs::CameraInfo& msg);
  void initRectification_(const sensor_msgs::CameraInfo& msg,
                          cv::Mat& map1, cv::Mat& map2);
  void doRectify_(GPUNS::GpuMat& gpu_raw,
                  GPUNS::GpuMat& gpu_rect_color,
                  GPUNS::GpuMat& gpu_rect,
                  GPUNS::GpuMat& gpu_map1,
                  GPUNS::GpuMat& gpu_map2,
                  GPUNS::Stream& stream = GPUNS::Stream::Null());
};

}  // namespace

#endif  // PS4EYE_TEGRA_PS4EYE_PROC_HPP_
