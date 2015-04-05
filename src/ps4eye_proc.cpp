#include "ps4eye_tegra/ps4eye_proc.hpp"

namespace ps4eye_tegra {

void PS4EyeProc::onInit() {
  ros::NodeHandle& nh = getNodeHandle();
  ros::NodeHandle& private_nh = getPrivateNodeHandle();
  ros::NodeHandle nh_left_camera (nh, "/stereo/left");
  it_.reset(new image_transport::ImageTransport(nh));
  it_left_camera_.reset(new image_transport::ImageTransport(nh_left_camera));

  // gpu information
  ROS_ASSERT(cv::gpu::getCudaEnabledDeviceCount() > 0);
  cv::gpu::DeviceInfo info(cv::gpu::getDevice());

  // params
  std::string left_info_file;
  std::string right_info_file;
  std::string default_package_path = ros::package::getPath("ps4eye");
  private_nh.param(
      "camera_info_file_left",
      left_info_file,
      default_package_path + "/camera_info/left.yaml");
  readCameraInfo_(left_info_file, left_info_);
  initRectification_(left_info_, left_map1_, left_map2_);
  gpu_left_map1_.upload(left_map1_);
  gpu_left_map2_.upload(left_map2_);

  private_nh.param(
      "camera_info_file_right",
      right_info_file,
      default_package_path + "/camera_info/right.yaml");
  readCameraInfo_(right_info_file, right_info_);
  initRectification_(right_info_, right_map1_, right_map2_);
  gpu_right_map1_.upload(right_map1_);
  gpu_right_map2_.upload(right_map2_);

  // left_model_.fromCameraInfo(left_info_);
  // right_model_.fromCameraInfo(right_info_);
  stereo_model_.fromCameraInfo(left_info_, right_info_);

  l_x_offset_ = 688;
  l_y_offset_ = 0;
  l_width_ = 640;
  l_height_ = 400;
  r_x_offset_ = 48;
  r_y_offset_ = 0;
  r_width_ = 640;
  r_height_ = 400;

  disparity_.create(cv::Size(l_width_, l_height_), CV_8UC1);
  left_cvimage_.image.create(cv::Size(l_width_, l_height_), CV_8UC3);

  // block_matcher_.preset = cv::gpu::StereoBM_GPU::BASIC_PRESET;
  block_matcher_.preset = cv::gpu::StereoBM_GPU::PREFILTER_XSOBEL;
  block_matcher_.ndisp = 64;
  block_matcher_.winSize = 15;
  block_matcher_.avergeTexThreshold = 10.0;

  ros::SubscriberStatusCallback connect_cb =
      boost::bind(&PS4EyeProc::connectCallback, this);
  image_transport::SubscriberStatusCallback connect_cb_image =
      boost::bind(&PS4EyeProc::connectCallback, this);
  pub_disparity_ =
      nh.advertise<DisparityImage>("/stereo/disparity", 1,
                                   connect_cb, connect_cb);
  pub_left_camera_ =
      it_left_camera_->advertiseCamera("image_rect_color", 1,
                                       connect_cb_image, connect_cb_image,
                                       connect_cb, connect_cb);
  pub_right_info_ =
      nh.advertise<CameraInfo>("/stereo/right/camera_info", 1,
                               connect_cb, connect_cb);

  approximate_sync_.reset(
      new ApproximateSync(
          ApproximatePolicy(10),
          sub_image_, sub_info_));
  approximate_sync_->registerCallback(
      boost::bind(&PS4EyeProc::imageCallback, this, _1, _2));
}

void PS4EyeProc::connectCallback() {
  boost::lock_guard<boost::mutex> lock(connection_mutex_);
  if (pub_disparity_.getNumSubscribers() == 0 &&
      pub_left_camera_.getNumSubscribers() == 0) {
    sub_image_.unsubscribe();
    sub_info_ .unsubscribe();
  } else if (!sub_image_.getSubscriber()) {
    ros::NodeHandle& nh = getNodeHandle();
    image_transport::TransportHints hints("raw",
                                          ros::TransportHints(),
                                          getPrivateNodeHandle());
    sub_image_.subscribe(*it_, "/camera/image_raw", 1, hints);
    sub_info_ .subscribe(nh, "/camera/camera_info", 1);
  }
}

void PS4EyeProc::imageCallback(const ImageConstPtr& image_msg,
                               const CameraInfoConstPtr& info_msg) {
  ROS_INFO("start proc");
  const cv::Mat input =
      cv_bridge::toCvShare(image_msg, sensor_msgs::image_encodings::BGR8)->image;

  // crop left and right images
  cv::Mat left_raw =
      input(cv::Rect(l_x_offset_, l_y_offset_, l_width_, l_height_));
  cv::Mat right_raw =
      input(cv::Rect(r_x_offset_, r_y_offset_, r_width_, r_height_));

  ROS_INFO("rectify");
  // rectify
  cv::Mat left_rect_color, left_rect;
  cv::gpu::GpuMat gpu_left_raw, gpu_left_rect_color, gpu_left_rect;
  gpu_left_raw.upload(left_raw);
  doRectify_(gpu_left_raw, gpu_left_rect_color, gpu_left_rect,
             gpu_left_map1_, gpu_left_map2_);

  cv::Mat right_rect_color, right_rect;
  cv::gpu::GpuMat gpu_right_raw, gpu_right_rect_color, gpu_right_rect;
  gpu_right_raw.upload(right_raw);
  doRectify_(gpu_right_raw, gpu_right_rect_color, gpu_right_rect,
             gpu_right_map1_, gpu_right_map2_);


  ROS_INFO("stereo matching");
  // cv::gpu::GpuMat gpu_left_rect, gpu_right_rect;
  // gpu_left_rect.upload(left_rect);
  // gpu_right_rect.upload(right_rect);
  cv::gpu::GpuMat gpu_disp;
  cv::gpu::Stream gpu_stream;
  ROS_INFO(" stereoBM in");
  block_matcher_(gpu_left_rect, gpu_right_rect, gpu_disp, gpu_stream);
  ROS_INFO(" stereoBM out");
  //gpu_disp.download(disparity_);
  gpu_stream.enqueueDownload(gpu_disp, disparity_);
  gpu_stream.enqueueDownload(gpu_left_rect_color, left_cvimage_.image);

  // cpu proc
  // stereo matching
  // from image_pipeline/stereo_image_proc
  static const int DPP = 8; // disparities per pixel
  static const double inv_dpp = 1.0 / DPP;

  // Allocate new disparity image message
  DisparityImagePtr disp_msg = boost::make_shared<DisparityImage>();
  disp_msg->header         = info_msg->header;
  disp_msg->image.header   = info_msg->header;

  // Compute window of (potentially) valid disparities
  int border   = block_matcher_.winSize / 2;
  int left   = block_matcher_.ndisp + 0 + border - 1;
  int wtf = border + 0;
  int right  = l_width_ - 1 - wtf;
  int top    = border;
  int bottom = l_height_ - 1 - border;
  disp_msg->valid_window.x_offset = left;
  disp_msg->valid_window.y_offset = top;
  disp_msg->valid_window.width    = right - left;
  disp_msg->valid_window.height   = bottom - top;


  sensor_msgs::Image& dimage = disp_msg->image;
  dimage.height = l_height_;
  dimage.width = l_width_;
  dimage.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
  dimage.step = dimage.width * sizeof(float);
  dimage.data.resize(dimage.step * dimage.height);
  cv::Mat_<float> dmat(dimage.height, dimage.width,
                       (float*)&dimage.data[0], dimage.step);

  left_cvimage_.header = info_msg->header;
  left_cvimage_.encoding = sensor_msgs::image_encodings::BGR8;

  left_info_.header = info_msg->header;
  right_info_.header = info_msg->header;

  // Stereo parameters
  disp_msg->f = stereo_model_.right().fx();
  disp_msg->T = stereo_model_.baseline();

  /// @todo Window of (potentially) valid disparities

  // Disparity search range
  disp_msg->min_disparity = 0;
  disp_msg->max_disparity = disp_msg->min_disparity + block_matcher_.ndisp - 1;
  disp_msg->delta_d = inv_dpp;



  // wait for gpu process completion
  gpu_stream.waitForCompletion();

  // We convert from fixed-point to float disparity and also adjust for any x-offset between
  // the principal points: d = d_fp*inv_dpp - (cx_l - cx_r)
  disparity_.convertTo(dmat, dmat.type(), inv_dpp,
                       -(stereo_model_.left().cx() -
                         stereo_model_.right().cx()));
  ROS_ASSERT(dmat.data == &dimage.data[0]);
  /// @todo is_bigendian? :)

  // Adjust for any x-offset between the principal points: d' = d - (cx_l - cx_r)
  double cx_l = stereo_model_.left().cx();
  double cx_r = stereo_model_.right().cx();
  if (cx_l != cx_r) {
    cv::Mat_<float> disp_image(disp_msg->image.height, disp_msg->image.width,
                              reinterpret_cast<float*>(&disp_msg->image.data[0]),
                              disp_msg->image.step);
    cv::subtract(disp_image, cv::Scalar(cx_l - cx_r), disp_image);
  }

  pub_disparity_.publish(disp_msg);

  sensor_msgs::ImagePtr left_image_msg = left_cvimage_.toImageMsg();
  sensor_msgs::CameraInfoPtr left_info_msg =
      boost::make_shared<sensor_msgs::CameraInfo>(left_info_);
  sensor_msgs::CameraInfoPtr right_info_msg =
      boost::make_shared<sensor_msgs::CameraInfo>(right_info_);

  pub_left_camera_.publish(left_image_msg, left_info_msg);
  pub_right_info_.publish(right_info_msg);

  ROS_INFO("end porc");
}

void PS4EyeProc::readCameraInfo_(const std::string& filename,
                                 sensor_msgs::CameraInfo& msg) {
  YAML::Node info = YAML::LoadFile(filename);
  msg.height = info["image_height"].as<int>();
  msg.width = info["image_width"].as<int>();

  msg.distortion_model = info["distortion_model"].as<std::string>();

  msg.D.clear();
  YAML::Node info_d = info["distortion_coefficients"]["data"];
  for (size_t i = 0; i < info_d.size(); i++) {
    msg.D.push_back(info_d[i].as<double>());
  }

  YAML::Node info_k = info["camera_matrix"]["data"];
  for (size_t i = 0; i < info_k.size(); i++) {
    msg.K[i] = info_k[i].as<double>();
  }

  YAML::Node info_r = info["rectification_matrix"]["data"];
  for (size_t i = 0; i < info_r.size(); i++) {
    msg.R[i] = info_r[i].as<double>();
  }

  YAML::Node info_p = info["projection_matrix"]["data"];
  for (size_t i = 0; i < info_p.size(); i++) {
    msg.P[i] = info_p[i].as<double>();
  }

  msg.binning_x = info["binning_x"].as<int>();
  msg.binning_y = info["binning_y"].as<int>();

  YAML::Node info_roi = info["roi"];
  msg.roi.x_offset = info_roi["x_offset"].as<int>();
  msg.roi.y_offset = info_roi["y_offset"].as<int>();
  msg.roi.width = info_roi["width"].as<int>();
  msg.roi.height = info_roi["height"].as<int>();
  msg.roi.do_rectify = info_roi["do_rectify"].as<bool>();
}

void PS4EyeProc::initRectification_(const sensor_msgs::CameraInfo& msg,
                                    cv::Mat& map1, cv::Mat& map2) {
  cv::Mat_<double> cv_D(1, msg.D.size());
  for (size_t i = 0; i < msg.D.size(); i++) {
    cv_D(i) = msg.D[i];
  }

  cv::Matx33d cv_K(&msg.K[0]);
  cv::Matx33d cv_R(&msg.R[0]);
  cv::Matx34d cv_P(&msg.P[0]);

  cv::initUndistortRectifyMap(cv_K, cv_D, cv_R, cv_P,
                              cv::Size(msg.width, msg.height),
                              CV_32FC1, map1, map2);
}

void PS4EyeProc::doRectify_(
    cv::gpu::GpuMat& gpu_raw,
    cv::gpu::GpuMat& gpu_rect_color,
    cv::gpu::GpuMat& gpu_rect,
    cv::gpu::GpuMat& gpu_map1,
    cv::gpu::GpuMat& gpu_map2) {
  // cv::remap(raw, rect_color, map1, map2, cv::INTER_LINEAR);
  // cv::cvtColor(rect_color, rect, CV_BGR2GRAY);
  cv::gpu::remap(gpu_raw, gpu_rect_color,
                 gpu_map1, gpu_map2, cv::INTER_LINEAR);
  cv::gpu::cvtColor(gpu_rect_color, gpu_rect, CV_BGR2GRAY);
}

}  // namespace

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(ps4eye_tegra::PS4EyeProc, nodelet::Nodelet)
