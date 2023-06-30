
#pragma once

#ifndef DATA_STRUCTURE_H
#define DATA_STRUCTURE_H

#include <Common.h>
#include <Config.h>

typedef Eigen::Matrix4d SE3;
typedef Eigen::Matrix3d SO3;

struct Frame;
struct MapPoint;

struct Feature
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    typedef std::shared_ptr<Feature> Ptr;
    std::weak_ptr<Frame> frame_;
    int id_;
    cv::KeyPoint position_;
    std::vector<double> rgb_;
    double depth_ = 1.0;
    std::weak_ptr<MapPoint> map_point_;
    bool is_outlier_ = false;
    bool isAssociated_ = false;

    Feature() {}
    Feature(std::shared_ptr<Frame> frame, const cv::KeyPoint &kp, double depth) : frame_(frame), position_(kp), depth_(depth) {}
    Feature(int id,std::shared_ptr<Frame> frame, const cv::KeyPoint &kp, double depth, bool isAssociated) : id_(id), frame_(frame), position_(kp), depth_(depth), isAssociated_(isAssociated) {}
    Feature(std::shared_ptr<Frame> frame, const cv::KeyPoint &kp, double depth, std::vector<double> rgb) : frame_(frame), position_(kp), depth_(depth), rgb_(rgb) {}
    Feature(int id, std::shared_ptr<Frame> frame, const cv::KeyPoint &kp, double depth, std::vector<double> rgb) : id_(id), frame_(frame), position_(kp), depth_(depth), rgb_(rgb) {}
};

struct Frame
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Frame> Ptr;

    unsigned long id_ = 0;          // id of this frame
    unsigned long keyframe_id_ = 0; // id of key frame
    bool is_keyframe_ = false;      // is this frame keyframe?
    double time_stamp_;             // time stamp
    SE3 pose_;                      // Tcw Pose
    std::mutex pose_mutex_;         // Pose lock
    cv::Mat rgb_;                   // RGB image
    cv::Mat gray_;                  // Gray image
    cv::Mat depth_;                 // Pseudo depth image
    Mat33 K_;                       // Intrinsic
    bool depthAvailable = false;
    cv::Mat descriptors_;
    std::vector<cv::Mat> imgPyramid_;

    std::vector<std::shared_ptr<Feature>> features_;

    // triangle pathces

public: // data members
    Frame(long id, double time_stamp, const SE3 &pose, const cv::Mat &rgb, const Mat33 &K)
    {
        id_ = id;
        time_stamp_ = time_stamp;
        pose_ = pose;
        rgb_ = rgb;
        cv::cvtColor(rgb_, gray_, cv::COLOR_RGB2GRAY);
        K_ = K;
    }

    Frame(long id, double time_stamp, const SE3 &pose, const cv::Mat &rgb, const cv::Mat &depth, const Mat33 &K)
    {
        id_ = id;
        time_stamp_ = time_stamp;
        pose_ = pose;
        rgb_ = rgb;
        cv::cvtColor(rgb_, gray_, cv::COLOR_RGB2GRAY);
        depth_ = depth;
        K_ = K;
        depthAvailable = true;
    }

    Frame(long id, double time_stamp, const SE3 &pose, const cv::Mat &rgb, const cv::Mat &depth, const Mat33 &K, const int img_width, const int img_height)
    {
        id_ = id;
        time_stamp_ = time_stamp;
        pose_ = pose;
        rgb_ = rgb;
        cv::cvtColor(rgb_, gray_, cv::COLOR_RGB2GRAY);
        depth_ = depth;
        K_ = K;
        depthAvailable = true;
    }

    // set and get pose, thread safe
    SE3 Pose()
    {
        std::unique_lock<std::mutex> lck(pose_mutex_);
        return pose_;
    }

    void setPose(const SE3 &pose)
    {
        std::unique_lock<std::mutex> lck(pose_mutex_);
        pose_ = pose;
    }

    // Set up keyframes, allocate and keyframe id
    void setKeyFrame()
    {
        static long keyframe_factory_id = 0;
        is_keyframe_ = true;
        keyframe_id_ = keyframe_factory_id++;
    }

    // coordinate transform: world, camera, pixel
    Vec3 world2camera(const Vec3 &p_w)
    {
        return pose_.block<3, 3>(0, 0) * p_w + pose_.block<3, 1>(0, 3);
    }
    Vec3 camera2world(const Vec3 &p_c)
    {
        SE3 T_wc = pose_.inverse();
        return T_wc.block<3, 3>(0, 0) * p_c + T_wc.block<3, 1>(0, 3);
    }
    Vec2 camera2pixel(const Vec3 &p_c)
    {
        // Pinhole
        // return Vec2(K_(0, 0) * p_c(0) / p_c(2) + K_(0, 2),
        //             K_(1, 1) * p_c(1) / p_c(2) + K_(1, 2));

        // Omni-directional camera

        int width = gray_.cols;
        int height = gray_.rows;

        double px = p_c[0];
        double py = p_c[1];
        double pz = p_c[2];

        double norm = std::sqrt(px * px + py * py + pz * pz);

        // Normalize the coordinates
        px /= norm;
        py /= norm;
        pz /= norm;

        double phi = std::asin(py);
        double theta;

        if (px > 0 && pz > 0)
            theta = std::asin(px / std::cos(phi));
        else if (px > 0 && pz < 0)
            theta = M_PI - std::asin(px / std::cos(phi));
        else if (px < 0 && pz > 0)
            theta = -std::asin(-px / std::cos(phi));
        else if (px < 0 && pz < 0)
            theta = std::asin(-px / std::cos(phi)) - M_PI;

        double u = (double)width / M_PI / 2 * theta + (double)width / 2;
        double v = (double)height / M_PI * phi + (double)height / 2;

        return Vec2(u, v);
    }

    double getZvalue(const Vec3 &p_w)
    {
        Vec3 p_c = world2camera(p_w);
        Vec3 p_img_h = K_ * p_c;
        return p_img_h(2);
    }


    Vec3 pixel2camera(const Vec2 &uv, double depth=1)
    {
        // Omni-directional camera

        int width = gray_.cols;
        int height = gray_.rows;

        // first, we need to convert the image coordinates back into the spherical coordinates
        double theta = (uv[0] - (double)width / 2) * (2 * M_PI / (double)width);
        double phi = (uv[1] - (double)height / 2) * (M_PI / (double)height);

        // next, we convert spherical coordinates back into Cartesian coordinates
        // assuming that the depth represents the radius in the spherical coordinate system
        double px = depth * std::sin(theta) * std::cos(phi);
        double py = depth * std::sin(phi);
        double pz = depth * std::cos(theta) * std::cos(phi);

        return Vec3(px, py, pz);
    }

    Vec3 pixel2world(const Vec2 &p_p, double depth = 1)
    {
        return camera2world(pixel2camera(p_p, depth));
    }

    Vec2 world2pixel(const Vec3 &p_w)
    {
        return camera2pixel(world2camera(p_w));
    }
};

struct MapPoint
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<MapPoint> Ptr;
    unsigned long id_ = 0; // ID
    bool is_outlier_ = false;
    Vec3 pos_ = Vec3::Zero(); // Position in world
    std::vector<double> rgb_;
    std::mutex data_mutex_;
    int observed_times_ = 0; // being observed by feature matching algo.
    std::vector<std::weak_ptr<Feature>> observations_;

    unsigned long id_frame_ = 0; // first observation frame id

    MapPoint() {}

    MapPoint(long id, Vec3 position);

    Vec3 Pos()
    {
        // share same mutex for protecting data
        std::unique_lock<std::mutex> lck(data_mutex_);
        return pos_;
    }

    void setPos(const Vec3 &pos)
    {
        // share same mutex for protecting data
        std::unique_lock<std::mutex> lck(data_mutex_);
        pos_ = pos;
    };

    void addObservation(std::shared_ptr<Feature> feature)
    {
        // share same mutex for protecting data
        std::unique_lock<std::mutex> lck(data_mutex_);
        observations_.push_back(feature);
        observed_times_++;
    }

    std::vector<std::weak_ptr<Feature>> GetObs()
    {
        // share same mutex for protecting data
        std::unique_lock<std::mutex> lck(data_mutex_);
        return observations_;
    }

    // factory function
    static MapPoint::Ptr createNewMappoint()
    {
        static long factory_id = 0;
        MapPoint::Ptr new_mappoint(new MapPoint);
        new_mappoint->id_ = factory_id++;
        return new_mappoint;
    }
};

class Map
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Map> Ptr;
    typedef std::unordered_map<unsigned long, MapPoint::Ptr> LandmarksType; // id and class (hash)
    typedef std::unordered_map<unsigned long, Frame::Ptr> KeyframesType;    // id and class (hash)

    Map(std::string config_path)
    {
        if (!Config::SetParameterFile(config_path))
            LOG(INFO) << "No configuration file loaded.";

        window_size_ = Config::Get<int>("window_size");
        min_dis_th_ = Config::Get<double>("keyframe_min_dist_threshold");
    }

    void insertKeyFrame(Frame::Ptr frame)
    {
        current_frame_ = frame;
        current_keyframe_ = frame;

        keyframe_id_.emplace_back(frame->id_);
        keyframes_.insert(make_pair(frame->id_, frame));
        active_keyframes_.insert(make_pair(frame->id_, frame));

        if (active_keyframes_.size() > window_size_)
        {
            // LOG(INFO) << "Remove old keyframe!!!";
            RemoveOldKeyframe();
        }
    }

    void insertMapPoint(MapPoint::Ptr map_point)
    {
        // LOG(INFO) << "Try Insert Map Point Function";
        if (landmarks_.find(map_point->id_) == landmarks_.end())
        {
            // New map point
            landmarks_.insert(make_pair(map_point->id_, map_point));
            active_landmarks_.insert(make_pair(map_point->id_, map_point));
        }
        else
        {
            // Exist but not active -> make it active
            if (active_landmarks_.find(map_point->id_) == active_landmarks_.end())
                active_landmarks_.insert(make_pair(map_point->id_, map_point));
        }
        // LOG(INFO) << "End Insert Map Point Function";
    }

    void RemoveOldKeyframe()
    {
        if (current_frame_ == nullptr)
            return;

        // // Find two frames closest to the current frame
        // double max_dis = 0, min_dis = 9999;
        // double max_kf_id = 0, min_kf_id = 0;
        // SE3 Twc = current_frame_->Pose().inverse(); // inverse for relative transform

        // for (auto &kf : active_keyframes_)
        // {
        //     if (kf.second == current_frame_)
        //         continue;

        //     Vec3 translation = (kf.second->Pose() * Twc).block<3, 1>(0, 3);
        //     // auto dis = (kf.second->Pose() * Twc).log().norm();
        //     double dis = translation.norm();
        //     if (dis > max_dis)
        //     {
        //         max_dis = dis;
        //         max_kf_id = kf.first;
        //     }
        //     if (dis < min_dis)
        //     {
        //         min_dis = dis;
        //         min_kf_id = kf.first;
        //     }
        // }

        // Frame::Ptr frame_to_remove = nullptr;

        // if (min_dis < min_dis_th_)
        // {
        //     // if distance between current keyframe and nearest one is close enough,
        //     // delete the nearest keyframe
        //     frame_to_remove = keyframes_.at(min_kf_id);
        // }
        // else
        // {
        //     // delete the furthest keyframe
        //     frame_to_remove = keyframes_.at(max_kf_id);
        // }

        // // LOG(INFO) << "remove keyframe " << frame_to_remove->id_;

        // // remove keyframe and related landmark observation

        // Just remove oldest keyframe?
        // TODO: No! vote using active landmarks
        // But first try removing oldest keyframe

        int min_kf_id = 100000;

        for (auto &kf : active_keyframes_)
        {
            int id = kf.first;
            if (id < min_kf_id)
                min_kf_id = id;
        }

        Frame::Ptr frame_to_remove = nullptr;
        frame_to_remove = keyframes_.at(min_kf_id);

        // // Check every active map points is activated on active keyframe
        // for (auto &lm : active_landmarks_)
        // {
        //     auto mp = lm.second;
        //     for (auto &obs : mp->observations_)
        //     {

        //         auto feature = obs.lock();
        //         // TODO: Why nullptr feature exist?
        //         if (feature == nullptr)
        //             continue;
        //         auto frame = feature->frame_.lock();
        //         int frame_id = frame->id_;
        //         if (keyframes_.find(frame_id) != keyframes_.end())
        //         {
        //             if (active_keyframes_.find(frame_id) == active_keyframes_.end())
        //             {

        //                 LOG(INFO) << "!!!!!!!! Fatal Error: active landmarks is not observed in active keyframe";
        //                 LOG(INFO) << "FRAME ID: " << frame_id;
        //                 LOG(INFO) << "####### LIST OF KEYFRAME IDS";
        //                 for (auto &kf : keyframes_)
        //                 {
        //                     LOG(INFO) << "ID: " << kf.second->id_;
        //                 }
        //                 LOG(INFO) << "####### LIST OF ACTIVE KEYFRAME IDS";
        //                 for (auto &kf : active_keyframes_)
        //                 {
        //                     LOG(INFO) << "ID: " << kf.second->id_;
        //                 }
        //             }
        //         }
        //     }
        // }

        // Map Clean

        for (auto feat : frame_to_remove->features_)
        {
            if (feat == nullptr)
                continue;

            auto mp = feat->map_point_.lock();
            if (mp)
            {
                // Delete observation related frame_to_remove

                // Eliminate invalid observation (but, I don't know why nullptr observation created.)
                std::vector<std::weak_ptr<Feature>> observations;

                for (size_t i = 0; i < mp->observations_.size(); i++)
                {
                    auto feature = mp->observations_[i].lock();

                    if (feature != nullptr)
                    {
                        if (feature != feat)
                            observations.push_back(mp->observations_[i]);
                    }
                }
                mp->observations_ = observations;
                mp->observed_times_ = (int)observations.size();

                if (mp->observations_.size() == 0)
                    active_landmarks_.erase(mp->id_);
            }
        }

        active_keyframes_.erase(frame_to_remove->id_);
    }

    LandmarksType GetAllMapPoints()
    {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return landmarks_;
    }
    KeyframesType GetAllKeyFrames()
    {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return keyframes_;
    }

    int window_size_;
    double min_dis_th_ = 0.1;
    std::mutex data_mutex_;
    LandmarksType landmarks_;        // all landmarks
    LandmarksType active_landmarks_; // active landmarks
    KeyframesType keyframes_;        // all keyframes
    KeyframesType active_keyframes_; // all keyframes
    Frame::Ptr current_keyframe_;    // current keyframe
    Frame::Ptr current_frame_ = nullptr;
    std::vector<int> keyframe_id_;
};

static Eigen::Vector3d triangulation(const Eigen::Matrix4d T_cw_1,
                                     const Eigen::Matrix4d T_cw_2,
                                     const Eigen::Vector2d xy_1,
                                     const Eigen::Vector2d xy_2,
                                     const Eigen::Matrix3d K_1,
                                     const Eigen::Matrix3d K_2)
{

    Eigen::Matrix<double, 3, 4> P1, P2;

    Eigen::Vector3d x_1(xy_1[0], xy_1[1], 1);
    Eigen::Vector3d x_2(xy_2[0], xy_2[1], 1);

    P1.block<3, 3>(0, 0) = K_1 * T_cw_1.block<3, 3>(0, 0);
    P1.block<3, 1>(0, 3) = K_1 * T_cw_1.block<3, 1>(0, 3);

    P2.block<3, 3>(0, 0) = K_2 * T_cw_2.block<3, 3>(0, 0);
    P2.block<3, 1>(0, 3) = K_2 * T_cw_2.block<3, 1>(0, 3);

    Eigen::Matrix4d A;
    A.row(0) = x_1[1] * P1.row(2) - P1.row(1);
    A.row(1) = P1.row(0) - x_1[0] * P1.row(2);
    A.row(2) = x_2[1] * P2.row(2) - P2.row(1);
    A.row(3) = P2.row(0) - x_2[0] * P2.row(2);

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    Eigen::Vector4d X_hom = svd.matrixV().col(3);
    Eigen::Vector3d X(X_hom(0) / X_hom(3), X_hom(1) / X_hom(3), X_hom(2) / X_hom(3));

    return X;
}

#endif