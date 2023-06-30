#pragma once
#ifndef Tracker_H
#define Tracker_H

#include <DataStructure.h>
#include <ProjectionFactor.h>

class Tracker
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Tracker> Ptr;

    Tracker(std::string config_file_path_)
    {
        if (!Config::SetParameterFile(config_file_path_))
            LOG(INFO) << "No configuration file loaded.";

        if (Config::Get<std::string>("feature_type") == "ORB")
        {
            detector_ = cv::ORB::create(Config::Get<int>("num_features"));
            descriptor_ = cv::ORB::create(Config::Get<int>("num_features"));
            matcher_ORB = cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
        }
        else if (Config::Get<std::string>("feature_type") == "SIFT")
        {
            detector_ = cv::SIFT::create(Config::Get<int>("num_features"));
            descriptor_ = cv::SIFT::create(Config::Get<int>("num_features"));
            matcher_SIFT = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        }

        LOG(INFO) << "OpenCV Version: " << CV_VERSION;

        knn_ratio_ = Config::Get<double>("knn_ratio");
        num_features_ = Config::Get<int>("num_features");
        // match_save_dir_ = Config::Get<std::string>("match_save_dir");
        match_threshold_ = Config::Get<int>("match_threshold");

        image_width = Config::Get<int>("image_width");
        image_height = Config::Get<int>("image_height");
        match_save_dir_ = Config::Get<std::string>("match_save_dir");
        map_ = Map::Ptr(new Map(config_file_path_));

        min_distance = Config::Get<double>("min_distance_cam");
        max_distance = Config::Get<double>("max_distance_cam");
    }

    bool addFrame(Frame::Ptr frame);

    void setMap(Map::Ptr map)
    {
        map_ = map;
    }

    bool init();
    bool buildMap();
    bool insertKeyFrame();
    void setObservationsForKeyFrame();
    bool optimize();

    int detectFeatures();
    int matchTwoFrames(Frame::Ptr frame_dst,
                       std::vector<cv::DMatch> &matches,
                       std::vector<cv::KeyPoint> &kps1,
                       std::vector<cv::KeyPoint> &kps2,
                       bool ransac,
                       bool showMatch);

    Frame::Ptr current_frame_ = nullptr;
    Frame::Ptr reference_frame_ = nullptr;

    Map::Ptr map_ = nullptr;

    // params
    int num_features_ = 200;
    int match_threshold_ = 50;

    // image
    int image_width = 0;
    int image_height = 0;

    // utilities
    cv::Ptr<cv::FeatureDetector> detector_;       // feature detector in opencv
    cv::Ptr<cv::DescriptorExtractor> descriptor_; // feature descriptor extractor in opencv
    cv::Ptr<cv::DescriptorMatcher> matcher_SIFT;
    cv::FlannBasedMatcher matcher_ORB;
    double knn_ratio_ = 0.7;

    std::vector<Frame::Ptr> frames_;

    std::string match_save_dir_;

    double min_distance;
    double max_distance;
};

#endif