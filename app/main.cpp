
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream> // ifstream header
#include <string>  // getline header
#include <sstream>

#include <opencv2/imgcodecs.hpp>
#include <pangolin/pangolin.h>

#include <Viewer.h>
#include <Config.h>
#include <IOUtils.h>
#include <LidarHandler.h>
#include <pangolin/pangolin.h>
#include <ImuHandler.h>

#include <iostream>
#include <fstream>

/////////////////////////////////////////////////////////////////////////////////

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/StereoFactor.h>
#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <gtsam/geometry/SimpleCamera.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/ImuBias.h>
#include <cmath>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <DataStructure.h>
// #include <Tracker.h>
#include <GraphSolver.h>
#include <TrackerOmni.h>

using namespace gtsam;
using symbol_shorthand::X; // Lidar pose

/////////////////////////////////////////////////////////////////////////////////
int grid_id_g = 0;
cv::Scalar orange(0, 165, 255), blue(255, 0, 0), magneta(255, 0, 255), red(0, 0, 255);
double img_timestamp_curr;

gtsam::noiseModel::Diagonal::shared_ptr lidarMeasureNoise = gtsam::noiseModel::Isotropic::Sigma(6, 1e-4);
gtsam::noiseModel::Diagonal::shared_ptr cameraMeasureNoise = gtsam::noiseModel::Isotropic::Sigma(6, 1e-1);
gtsam::noiseModel::Diagonal::shared_ptr imuMeasureNoise = gtsam::noiseModel::Isotropic::Sigma(6, 1e-1);

bool graphInit = true;

Eigen::Quaterniond axisAngleToQuat(Eigen::Vector3d axis_angle)
{
    Eigen::Quaterniond quat(1, 0, 0, 0);
    double angle = axis_angle.norm();

    if (angle < 1e-50)
        return quat;

    double half_angle = angle / 2;
    quat = Eigen::Quaterniond(cos(half_angle),
                              sin(half_angle) / angle * axis_angle[0],
                              sin(half_angle) / angle * axis_angle[1],
                              sin(half_angle) / angle * axis_angle[2]);

    return quat;
}

Eigen::Vector3d quatToAxisAngle(Eigen::Quaterniond quat)
{
    Eigen::AngleAxisd axis_angle(quat);
    Eigen::Vector3d axis = axis_angle.axis();
    double angle = axis_angle.angle();
    return axis * angle;
}

std::vector<Eigen::Vector3f> xyz2uv_vectorized(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud, int width, int height)
{
    std::vector<Eigen::Vector3f> result;
    result.reserve(cloud->size());

    for (const auto &point : cloud->points)
    {
        float px = point.x;
        float py = point.y;
        float pz = point.z;
        float norm = std::sqrt(px * px + py * py + pz * pz);

        // Normalize the coordinates
        px /= norm;
        py /= norm;
        pz /= norm;

        float phi = std::asin(py);
        float theta;

        if (px > 0 && pz > 0)
            theta = std::asin(px / std::cos(phi));
        else if (px > 0 && pz < 0)
            theta = M_PI - std::asin(px / std::cos(phi));
        else if (px < 0 && pz > 0)
            theta = -std::asin(-px / std::cos(phi));
        else if (px < 0 && pz < 0)
            theta = std::asin(-px / std::cos(phi)) - M_PI;

        float u = (float)width / M_PI / 2 * theta + (float)width / 2;
        float v = (float)height / M_PI * phi + (float)height / 2;

        result.push_back(Eigen::Vector3f(u, v, norm));
    }

    return result;
}

int main(int argc, char **argv)
{

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgbCloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    ////////////////////// Load Parameters ///////////////////
    std::string config_path = "../config/config.yaml";
    Config::SetParameterFile(config_path);
    std::string lidar_folder = Config::Get<std::string>("lidar_folder");
    std::string image_folder = Config::Get<std::string>("image_folder");
    std::string imu_data_path = Config::Get<std::string>("imu_path");
    int id = Config::Get<int>("id");
    int last = Config::Get<int>("last");
    int init_id = id;
    int num_cumulated_cloud = Config::Get<int>("num_cumulated_cloud");
    double max_distance = Config::Get<double>("max_distance");
    double min_distance = Config::Get<double>("min_distance");

    ////////////////////// Read Dataset //////////////////////
    std::deque<ImuMessage> imu_queue;
    std::deque<LidarMessage> lidar_queue;
    std::deque<ImgMessage> img_queue;

    loadImuDataset(imu_data_path, imu_queue);
    loadLaserDataset(lidar_folder, lidar_queue);
    loadImageDataset(image_folder, img_queue);

    LOG(INFO) << "######## Data Loaded ########";
    LOG(INFO) << "# of imu sequences: " << imu_queue.size();
    LOG(INFO) << "# of lidar sequences: " << lidar_queue.size();
    LOG(INFO) << "# of img sequences: " << img_queue.size();

    std::vector<double> timestamps_fused;
    std::vector<int> msg_types;

    // Projection Test
    Eigen::Matrix4d T_cl_right, T_cl_back, T_cl_front, T_cc;

    //             [[ 0.99979437 -0.00477388  0.01970845  0.085044  ]
    //  [ 0.0047183   0.99998476  0.00286589 -0.06786   ]
    //  [-0.01972183 -0.00277231  0.99980166  0.120855  ]
    //  [ 0.          0.          0.          1.        ]]

    T_cc << 0.99979437, -0.00477388, 0.01970845, 0.085044,
        0.0047183, 0.99998476, 0.00286589, -0.06786,
        -0.01972183, -0.00277231, 0.99980166, 0.120855,
        0, 0, 0, 1;
    T_cl_right << -0.999339, -0.00260931, 0.0362488, -0.24700593,
        -0.0363188, 0.0355867, -0.998706, 0.57614229,
        0.00131596, -0.999363, -0.035658, -0.19790975,
        0, 0, 0, 1;

    T_cl_right = T_cc * T_cl_right;

    Eigen::Matrix4d y90;
    y90 << 0, 0, 1, 0,
        0, 1, 0, 0,
        -1, 0, 0, 0,
        0, 0, 0, 1;

    Eigen::Matrix4d y180;
    y180 << -1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, -1, 0,
        0, 0, 0, 1;

    Eigen::Matrix4d y270;
    y270 << 0, 0, -1, 0,
        0, 1, 0, 0,
        1, 0, 0, 0,
        0, 0, 0, 1;

    T_cl_back = y90.inverse() * T_cl_right;
    T_cl_front = y180 * T_cl_back;

    Eigen::Matrix4d T_cl = T_cl_front;

    // flag: imu(0) / lidar(1) / image(2)
    for (size_t imuId = 0; imuId < imu_queue.size(); imuId++)
    {
        timestamps_fused.push_back(imu_queue[imuId].timestamp);
        msg_types.push_back(0);
    }

    for (size_t lidarId = 0; lidarId < lidar_queue.size(); lidarId++)
    {
        timestamps_fused.push_back(lidar_queue[lidarId].timestamp);
        msg_types.push_back(1);
    }

    for (size_t imgId = 0; imgId < img_queue.size(); imgId++)
    {
        timestamps_fused.push_back(img_queue[imgId].timestamp);
        msg_types.push_back(2);
    }

    std::vector<size_t> order = argsort_d(timestamps_fused);

    ////////////////////// Extrinsics ////////////////////////
    // Eigen::Matrix4d T_wl = Eigen::Matrix4d::Identity();
    // Eigen::Matrix4d T_il;
    // T_il << -0.99999353, 0.00169655, 0.00327721, -0.0554467, 0.00172166, 0.99996872, 0.00767292, 0.00517844, -0.00326409, 0.0076785, -0.99996512, 0.03128449, 0, 0, 0, 1;
    // Eigen::Matrix4d T_ic;
    // T_ic << -0.00700046, 0.00632448, -0.9999555, -0.02536939, -0.99989, -0.01312013, 0.00691702, 0.02244737, -0.0130758, 0.99989393, 0.00641563, -0.02027493, 0, 0, 0, 1;
    // Eigen::Matrix3d K;
    // K << 872.614, 0, 615.868, 0, 872.992, 552.648, 0.0, 0.0, 1.0;
    // Eigen::Matrix4d T_cl = T_ic.inverse() * T_il;
    // Eigen::Matrix4d T_li = T_il.inverse();

    // Eigen::Matrix4d T_wi = Eigen::Matrix4d::Identity();

    ////////////////////// Init Viewer ////////////////////////
    Viewer::Ptr viewer = Viewer::Ptr(new Viewer);
    viewer->initialize();
    viewer->id = id;
    bool initViewer = false;

    double imu_time_last = -1;
    int imu_frequency = 100;
    int imu_init_cnt_max = 500; // 5 seconds for initialization

    std::vector<Eigen::Matrix4d> trajectory;
    std::vector<double> timestamp_vec;

    // Initial Values
    // bool imuInit = false;
    bool cameraInit = false;
    // bool lidarInit = false;

    int imu_cnt = 0;
    Eigen::Vector3d mean_acc(0, 0, 0);

    LidarHandler lidarHandler(config_path);
    lidarHandler.T_wl = Eigen::Matrix4d::Identity(); // * T_il.inverse();

    ImuHandler imuHandler;
    // Tracker::Ptr tracker(new Tracker(config_path));

    TrackerOmni::Ptr tracker(new TrackerOmni(config_path));
    Map::Ptr map(new Map(config_path));

    tracker->setMap(map);

    GraphSolver graphSolver;

    viewer->width = tracker->image_width;
    viewer->height = tracker->image_height;
    viewer->fx = 480;
    viewer->fy = 480;
    viewer->cx = 480;
    viewer->cy = 480;

    cv::Mat img_ref;
    cv::Mat img_curr;

    double img_time_ref, lidar_time_ref;
    double img_time_curr, lidar_time_curr;

    std::deque<ImuMessage> imu_msg_queue;
    std::deque<ImuMessage> imu_msg_queue_cam;

    double minimum_gap_camera = 0.1;
    double minimum_gap_lidar = 0.05;

    int frameId = 0;

    pcl::PointCloud<PointType>::Ptr currCloud(new pcl::PointCloud<PointType>);

    std::deque<pcl::PointCloud<PointType>::Ptr> cumulatedCloudWorld;

    /////////////////////////////////////////////////////////////////////////////////////

    // For optical flow

    /////////////////////////////////////////////////////////////////////////////////////

    cv::Mat prevFrame, currFrame, prevGray, currGray;
    std::vector<cv::Point2f> prevPts, currPts;

    double qualityLevel = 0.1;
    double minDistance = 30;
    int blockSize = 3, gradientSize = 3;
    bool useHarrisDetector = false;
    double k = 0.04;
    bool imgInit = false;

    // Create mask with same size as image, initially all zeros (black)
    cv::Mat maskGlobal = cv::Mat::zeros(cv::Size(1920, 960), CV_8U);

    // Set middle half of the mask to white (255)
    maskGlobal.rowRange(960 / 4, 960 * 3 / 4).setTo(cv::Scalar(255));

    std::vector<cv::Mat> prevPyramid, currPyramid;
    cv::Size winSize(31, 31);
    int maxLevel = 4;

    for (size_t msgId = (size_t)id; msgId < msg_types.size(); msgId++)
    {

        // if (cameraInit)
        // {
        //     std::vector<Frame::Ptr> keyframes_viz;
        //     for (auto &frame_set : tracker->map_->active_keyframes_)
        //     {
        //         auto key_frame = frame_set.second;
        //         keyframes_viz.push_back(key_frame);
        //     }

        //     viewer->keyframes = keyframes_viz;
        //     viewer->curr_frame = tracker->current_frame_;
        // }

        // // LOG(INFO) << "# of active landmarks: " << tracker->map_->active_landmarks_.size();
        // // LOG(INFO) << "# of active keyframes: " << tracker->map_->active_keyframes_.size();

        // cloudXYZ::Ptr cumulated_cloud2(new cloudXYZ);

        // for (auto &landmark : tracker->map_->active_landmarks_)
        // {
        //     auto mp = landmark.second;
        //     // LOG(INFO) <<"Pos: "<<mp->pos_.transpose();
        //     cumulated_cloud2->push_back(pcl::PointXYZ(mp->pos_[0], mp->pos_[1], mp->pos_[2]));
        // }

        // viewer->cloud_cumulated = cumulated_cloud2;

        // if (msg_types[order[msgId]] == 0)
        // {
        //     // Imu message callback
        //     ImuMessage imu_msg = imu_queue.front();

        //     if (!imuHandler.imuInit)
        //     {
        //         if (imu_cnt < imu_init_cnt_max)
        //         {
        //             // Initialization
        //             mean_acc += (imu_msg.linear_acc / (double)imu_init_cnt_max);
        //             imu_cnt += 1;
        //             imu_queue.pop_front();
        //             continue;
        //         }
        //         else
        //         {
        //             imuHandler.imuInit = true;
        //             imuHandler.imuMeanAcc = mean_acc;
        //             // graphSolver.imuMeanAcc = mean_acc;
        //         }
        //     }

        //     else
        //     {
        //         imu_msg_queue_cam.push_back(imu_queue.front());
        //         imu_msg_queue.push_back(imu_queue.front());
        //         imu_queue.pop_front();
        //     }
        // }

        if (msg_types[order[msgId]] == 1)
        {
            LidarMessage lidar_msg = lidar_queue.front(); // current image msg

            // LOG(INFO) << std::setprecision(18) << "Timestamp(LidarScan): " << lidar_msg.timestamp;

            if (!lidarHandler.lidarInit)
            {
                pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>);
                readLaserScan(lidar_msg.lidar_path, cloud);
                lidarHandler.updateOdometry(cloud, false);
                lidar_time_ref = lidar_msg.timestamp;
                lidar_queue.pop_front();
                continue;
            }

            lidar_time_curr = lidar_msg.timestamp;

            // while (!imu_msg_queue.empty())
            // {
            //     ImuMessage imu_msg = imu_msg_queue.front();
            //     bool imu_too_old = (imu_msg.timestamp < lidar_time_ref);
            //     bool imu_over = (imu_msg.timestamp > lidar_time_curr);

            //     if (imu_too_old)
            //     {
            //         imu_msg_queue.pop_front();
            //         continue;
            //     }

            //     else if (imu_over)
            //     {
            //         break;
            //     }

            //     else
            //     {
            //         imu_vec.push_back(imu_msg);
            //         imu_msg_queue.pop_front();
            //     }
            // }

            pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>);
            readLaserScan(lidar_msg.lidar_path, cloud);

            // *currCloud = *cloud;

            // if (imu_vec.size() > 6)
            // {

            //     // imu preintegration
            //     Eigen::Matrix4d T_imu_rel;
            //     Eigen::Vector3d imuVelocity;
            //     imuHandler.integrateIMU(imu_vec, T_imu_rel, imuVelocity);
            //     imuHandler.T_wi_curr = imuHandler.T_wi_prev * T_imu_rel;

            //     Eigen::Matrix4d T_wl_init = imuHandler.T_wi_curr * T_il;

            //     deskewCloud(cloud, lidarHandler.T_wl, T_wl_init);

            //     lidarHandler.T_wl = T_wl_init; // check the difference
            // }
            // else
            //     std::cout << "No opt! // " << imu_vec.size() << std::endl;

            lidarHandler.updateOdometry(cloud, false);

            // imuHandler.T_wi_prev = lidarHandler.T_wl * T_li;

            // IMU Optimization
            // if (imu_vec.size() > 6)
            //     imuHandler.optimizeBias(lidarHandler.T_wl, T_il);
            // else
            //     imuHandler.resetIntegrator();

            if (cumulatedCloudWorld.size() > num_cumulated_cloud)
                cumulatedCloudWorld.pop_front();

            pcl::PointCloud<PointType>::Ptr cloudWorld(new pcl::PointCloud<PointType>);
            pcl::transformPointCloud(*cloud, *cloudWorld, lidarHandler.T_wl);

            cumulatedCloudWorld.push_back(cloudWorld);

            lidar_queue.pop_front();
            lidar_time_ref = lidar_time_curr;
        }

        else if (msg_types[order[msgId]] == 2)
        {
            ImgMessage img_msg = img_queue.front();
            img_queue.pop_front();
            // LOG(INFO) << std::setprecision(18) << "Timestamp(ImgERP): " << img_msg.timestamp;

            if (!lidarHandler.lidarInit || cumulatedCloudWorld.size() < num_cumulated_cloud)
                continue;

            bool isSyncWithLidar = false;
            // Time sync!
            if (abs(img_msg.timestamp - lidar_time_curr) < 0.02)
            {
                isSyncWithLidar = true;
            }

            cv::Mat imgERP = cv::imread(img_msg.img_path, 1);

            // LOG(INFO) << "Synced image! // " << imgERP.size();

            pcl::PointCloud<PointType>::Ptr nearCameraDenseCloud(new pcl::PointCloud<PointType>);
            pcl::PointCloud<PointType>::Ptr nearWorldDenseCloud(new pcl::PointCloud<PointType>);

            for (size_t k = 0; k < cumulatedCloudWorld.size(); k++)
            {
                *nearWorldDenseCloud += *cumulatedCloudWorld[k];
            }

            Eigen::Matrix4d T_cw = T_cl * lidarHandler.T_wl.inverse();

            if (isSyncWithLidar)
            {
                pcl::transformPointCloud(*nearWorldDenseCloud, *nearCameraDenseCloud, T_cw);
                std::vector<Eigen::Vector3f> uv_and_norm = xyz2uv_vectorized(nearCameraDenseCloud, 1920, 960);
                cv::Mat depth = cv::Mat::zeros(tracker->image_height, tracker->image_width, CV_32FC1);

                float maxDistance = 60.0;

                for (size_t uv_idx = 0; uv_idx < uv_and_norm.size(); uv_idx++)
                {
                    float distance = uv_and_norm[uv_idx][2];

                    if (distance < maxDistance)
                    {
                        int y = (int)uv_and_norm[uv_idx][1];
                        int x = (int)uv_and_norm[uv_idx][0];

                        if (y < 0 || y > tracker->image_height || x < 0 || x > tracker->image_width)
                            continue;

                        depth.at<float>((int)uv_and_norm[uv_idx][1], (int)uv_and_norm[uv_idx][0]) = distance;
                    }
                }

                Frame::Ptr frame(new Frame(frameId++, img_msg.timestamp, T_cw, cv::imread(img_msg.img_path), depth, Eigen::Matrix3d::Identity()));
                tracker->addFrame(frame);
            }
            else
            {
                Frame::Ptr frame(new Frame(frameId++, img_msg.timestamp, T_cw, cv::imread(img_msg.img_path), Eigen::Matrix3d::Identity()));
                tracker->addFrame(frame);
            }
        }

        // visualize map point

        auto active_landmarks = map->active_landmarks_;

        pcl::PointCloud<PointType>::Ptr cloud_feature(new pcl::PointCloud<PointType>);

        for (auto &lm : active_landmarks)
        {
            auto mp = lm.second;
            Vec3 xyz = mp->Pos();

            PointType p;
            p.x = xyz[0];
            p.y = xyz[1];
            p.z = xyz[2];

            cloud_feature->push_back(p);
        }

        viewer->cloud_map_local = lidarHandler.cloud_surf_map;
        viewer->cloud_map = lidarHandler.cloud_surf_map_global;
        viewer->cloud_surf = lidarHandler.cloud_surf_w;
        viewer->cloud_feature = cloud_feature;
        viewer->T_wl = lidarHandler.T_wl;
        viewer->spinOnce();
    }

    // std::ofstream ofile("/home/seungwon/Data/maxst/b2/trajectory.txt");

    // for (size_t i = 0; i < trajectory.size(); i++)
    // {
    //     double time = timestamp_vec[i];
    //     Eigen::Matrix3d R = trajectory[i].block<3, 3>(0, 0);
    //     Eigen::Quaterniond quat(R);
    //     Eigen::Vector3d trans = trajectory[i].block<3, 1>(0, 3);

    //     std::string line = "";
    //     line += (std::to_string(time) + " ");
    //     line += (std::to_string(trans[0]) + " ");
    //     line += (std::to_string(trans[1]) + " ");
    //     line += (std::to_string(trans[2]) + " ");
    //     line += (std::to_string(quat.x()) + " ");
    //     line += (std::to_string(quat.y()) + " ");
    //     line += (std::to_string(quat.z()) + " ");
    //     line += (std::to_string(quat.w()) + "\n");
    //     // std::cout << line;
    //     ofile << line;
    // }

    // ofile.close();

    return 0;
}
