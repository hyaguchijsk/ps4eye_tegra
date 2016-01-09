#include "ps4eye_tegra/ps4eye_proc.hpp"

namespace ps4eye_tegra {

void PS4EyeProc::onInit() {
  ros::NodeHandle& nh = getNodeHandle();
  ros::NodeHandle& private_nh = getPrivateNodeHandle();
  ros::NodeHandle nh_left_camera (nh, "/stereo/left");
  it_.reset(new image_transport::ImageTransport(nh));
  it_left_camera_.reset(new image_transport::ImageTransport(nh_left_camera));

  // gpu information
  ROS_ASSERT(GPUNS::getCudaEnabledDeviceCount() > 0);
  GPUNS::DeviceInfo info(GPUNS::getDevice());

  // params
  std::string left_info_file;
  std::string right_info_file;
  std::string default_package_path = ros::package::getPath("ps4eye");
  private_nh.param(
      "left_file_name",
      left_info_file,
      default_package_path + "/camera_info/left.yaml");
  readCameraInfo_(left_info_file, left_info_);
  initRectification_(left_info_, left_map1_, left_map2_);
  gpu_left_map1_.upload(left_map1_);
  gpu_left_map2_.upload(left_map2_);

  private_nh.param(
      "right_file_name",
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

  disparity_.create(cv::Size(l_width_, l_height_), CV_32FC1);
  left_rect_color_.create(cv::Size(l_width_, l_height_), CV_8UC3);
  left_cvimage_.image.create(cv::Size(l_width_, l_height_), CV_8UC3);

  private_nh.param("ndisparity", ndisp_, 96);
  private_nh.param("win_size", win_size_, 15);
  private_nh.param("filter_radius", filter_radius_, 3);
  private_nh.param("filter_iter", filter_iter_, 1);

  private_nh.param("stretch_factor", stretch_factor_, 2);

  private_nh.param("use_csbp", use_csbp_, true);
  private_nh.param("use_bilateral_filter", use_bilateral_filter_, true);
  private_nh.param("use_stretch", use_stretch_, false);

  if (!use_stretch_) {
    stretch_factor_ = 1;
  }

#if OPENCV3
  block_matcher_ = cv::cuda::createStereoBM(ndisp_ * stretch_factor_,
                                            win_size_);
  csbp_matcher_ =
      cv::cuda::createStereoConstantSpaceBP(ndisp_ * stretch_factor_,
                                            8, 4, 4, CV_16SC1);
  //block_matcher_->setPrefilterType(cv::StereoBM::PREFILTER_XSOBEL);
  //block_matcher_->setPrefilterCap(31);
  //block_matcher_->setTextureTheshold(10);
  bilateral_filter_ =
      cv::cuda::createDisparityBilateralFilter(ndisp_ * stretch_factor_,
                                               filter_radius_, filter_iter_);
#else
  block_matcher_.preset = cv::gpu::StereoBM_GPU::PREFILTER_XSOBEL;
  block_matcher_.ndisp = ndisp_;
  block_matcher_.winSize = win_size_;
  block_matcher_.avergeTexThreshold = 10.0;
#endif

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
  // static const int DPP = 16; // disparities per pixel
  // ^ means cv::StereoBM has 4 fractional bits,
  // so disparity = value / (2 ^ 4) = value / 16
  // in GPU version, disparity mat has raw disparity
  static const int DPP = stretch_factor_; // disparities per pixel
  static const double inv_dpp = 1.0 / DPP;

  ROS_INFO("start proc");
  GPUNS::Stream gpu_stream;
  const cv::Mat input =
      cv_bridge::toCvShare(image_msg, sensor_msgs::image_encodings::BGR8)->image;

  // crop left and right images
  // rectify
  ROS_INFO("  crop left");
  cv::Mat left_raw =
      input(cv::Rect(l_x_offset_, l_y_offset_, l_width_, l_height_));
  GPUNS::GpuMat gpu_left_raw, gpu_left_rect_color, gpu_left_rect;
  gpu_left_raw.upload(left_raw);
  ROS_INFO("  rectify left");
  doRectify_(gpu_left_raw, gpu_left_rect_color, gpu_left_rect,
             gpu_left_map1_, gpu_left_map2_, gpu_stream);

  ROS_INFO("  crop right");
  cv::Mat right_raw =
      input(cv::Rect(r_x_offset_, r_y_offset_, r_width_, r_height_));
  GPUNS::GpuMat gpu_right_raw, gpu_right_rect_color, gpu_right_rect;
  gpu_right_raw.upload(right_raw);
  ROS_INFO("  rectify right");
  doRectify_(gpu_right_raw, gpu_right_rect_color, gpu_right_rect,
             gpu_right_map1_, gpu_right_map2_, gpu_stream);

  // stretch
  GPUNS::GpuMat gpu_left_stretch;
  GPUNS::GpuMat gpu_right_stretch;
  if (use_stretch_) {
    ROS_INFO("  stretch left");
    cv::cuda::resize(gpu_left_rect, gpu_left_stretch,
                     cv::Size(stretch_factor_ * l_width_, l_height_),
                     0, 0, cv::INTER_LINEAR, gpu_stream);

    ROS_INFO("  stretch right");
    cv::cuda::resize(gpu_right_rect, gpu_right_stretch,
                     cv::Size(stretch_factor_ * r_width_, r_height_),
                     0, 0, cv::INTER_LINEAR, gpu_stream);
  }

  // wait for left and right image
  // gpu_stream.waitForCompletion();

  ROS_INFO(" stereo matching");
  // cv::gpu::GpuMat gpu_left_rect, gpu_right_rect;
  // gpu_left_rect.upload(left_rect);
  // gpu_right_rect.upload(right_rect);
  GPUNS::GpuMat gpu_disp, gpu_disp_filtered, gpu_disp_stretch;
  ROS_INFO("  stereoBM in");
#if OPENCV3
  if (use_stretch_) {
    if (use_csbp_) {
      csbp_matcher_->compute(gpu_left_stretch, gpu_right_stretch,
                             gpu_disp_stretch, gpu_stream);
    } else {
      block_matcher_->compute(gpu_left_stretch, gpu_right_stretch,
                              gpu_disp_stretch, gpu_stream);
    }
    if (use_bilateral_filter_) {
      bilateral_filter_->apply(gpu_disp_stretch, gpu_left_stretch,
                               gpu_disp_filtered, gpu_stream);
      cv::cuda::resize(gpu_disp_filtered, gpu_disp,
                       cv::Size(l_width_, l_height_),
                       0, 0, cv::INTER_LINEAR, gpu_stream);
    } else {
      cv::cuda::resize(gpu_disp_stretch, gpu_disp,
                       cv::Size(l_width_, l_height_),
                       0, 0, cv::INTER_LINEAR, gpu_stream);
    }
    gpu_disp.download(disparity_, gpu_stream);
    // gpu_disp_filtered.download(disparity_, gpu_stream);
  } else {
    if (use_csbp_) {
      csbp_matcher_->compute(gpu_left_rect, gpu_right_rect,
                             gpu_disp, gpu_stream);
    } else {
      block_matcher_->compute(gpu_left_rect, gpu_right_rect,
                              gpu_disp, gpu_stream);
    }
    if (use_bilateral_filter_) {
      bilateral_filter_->apply(gpu_disp, gpu_left_rect, gpu_disp_filtered,
                               gpu_stream);
      gpu_disp_filtered.download(disparity_, gpu_stream);
    } else {
      gpu_disp.download(disparity_, gpu_stream);
    }
  }
  gpu_left_rect_color.download(left_rect_color_, gpu_stream);
#else
  block_matcher_(gpu_left_rect, gpu_right_rect, gpu_disp, gpu_stream);
  gpu_stream.enqueueDownload(gpu_left_rect_color, left_cvimage_.image);
  gpu_stream.enqueueDownload(gpu_disp, disparity_);
#endif
  ROS_INFO("  stereoBM out");
  //gpu_disp.download(disparity_);

  // cpu proc
  // stereo matching
  // from image_pipeline/stereo_image_proc

  // Allocate new disparity image message
  DisparityImagePtr disp_msg = boost::make_shared<DisparityImage>();
  disp_msg->header         = info_msg->header;
  disp_msg->image.header   = info_msg->header;

  // Compute window of (potentially) valid disparities
  int border   = win_size_ / 2;
  int left   = ndisp_ + 0 + border - 1;
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
  disp_msg->max_disparity = disp_msg->min_disparity + ndisp_ - 1;
  disp_msg->delta_d = inv_dpp;

  // wait for gpu process completion
  ROS_INFO(" GPU sync");
  gpu_stream.waitForCompletion();
  ROS_INFO(" GPU process end");

#if OPENCV3
  left_cvimage_.image = left_rect_color_.createMatHeader();
  ROS_INFO(" left image conversion end");
#endif

  // We convert from fixed-point to float disparity and also adjust for any x-offset between
  // the principal points: d = d_fp*inv_dpp - (cx_l - cx_r)
#if OPENCV3
  if (use_stretch_) {
    disparity_.createMatHeader().convertTo(
        dmat, dmat.type(), inv_dpp,
        -(stereo_model_.left().cx() -
          stereo_model_.right().cx()));
  } else {
    disparity_.createMatHeader().assignTo(dmat, dmat.type());
  }
#else
  disparity_.convertTo(dmat, dmat.type(), inv_dpp,
                       -(stereo_model_.left().cx() -
                         stereo_model_.right().cx()));
#endif
  ROS_ASSERT(dmat.data == &dimage.data[0]);
  /// @todo is_bigendian? :)
  ROS_INFO(" disparity conversion end");

  if (pub_disparity_.getNumSubscribers() > 0) {
    pub_disparity_.publish(disp_msg);
  }

  if (pub_left_camera_.getNumSubscribers() > 0) {
    sensor_msgs::ImagePtr left_image_msg = left_cvimage_.toImageMsg();
    sensor_msgs::CameraInfoPtr left_info_msg =
        boost::make_shared<sensor_msgs::CameraInfo>(left_info_);
    pub_left_camera_.publish(left_image_msg, left_info_msg);
  }

  sensor_msgs::CameraInfoPtr right_info_msg =
      boost::make_shared<sensor_msgs::CameraInfo>(right_info_);
  pub_right_info_.publish(right_info_msg);

  ROS_INFO("end proc");
}

void PS4EyeProc::readCameraInfo_(const std::string& filename,
                                 sensor_msgs::CameraInfo& msg) {
  ROS_INFO_STREAM("loading " << filename);

  YAML::Node info = YAML::LoadFile(filename);
  msg.height = info["image_height"].as<int>();
  msg.width = info["image_width"].as<int>();
  // ROS_INFO_STREAM("  dim: " << msg.width << " x " << msg.height);

  msg.distortion_model = info["distortion_model"].as<std::string>();
  // ROS_INFO_STREAM("  distortion: " << msg.distortion_model);

  msg.D.clear();
  YAML::Node info_d = info["distortion_coefficients"]["data"];
  for (size_t i = 0; i < info_d.size(); i++) {
    // ROS_INFO_STREAM("    " << info_d[i].as<double>());
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
    GPUNS::GpuMat& gpu_raw,
    GPUNS::GpuMat& gpu_rect_color,
    GPUNS::GpuMat& gpu_rect,
    GPUNS::GpuMat& gpu_map1,
    GPUNS::GpuMat& gpu_map2,
    GPUNS::Stream& stream) {
  // cv::remap(raw, rect_color, map1, map2, cv::INTER_LINEAR);
  // cv::cvtColor(rect_color, rect, CV_BGR2GRAY);
  GPUNS::remap(gpu_raw, gpu_rect_color,
               gpu_map1, gpu_map2, cv::INTER_LINEAR,
               cv::BORDER_CONSTANT, cv::Scalar(), stream);
  GPUNS::cvtColor(gpu_rect_color, gpu_rect, CV_BGR2GRAY, 0, stream);
}

}  // namespace

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(ps4eye_tegra::PS4EyeProc, nodelet::Nodelet)
