#pragma once
#ifndef Tracker_Omni_H
#define Tracker_Omni_H

#include <DataStructure.h>
#include <ProjectionFactor.h>

class TrackerOmni
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<TrackerOmni> Ptr;

    TrackerOmni(std::string config_file_path_)
    {
        if (!Config::SetParameterFile(config_file_path_))
            LOG(INFO) << "No configuration file loaded.";

        isInitialized = false;

        maxLevel = Config::Get<int>("maxLevelPyramid");
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

    bool Track();

    bool associateWithDepth(const cv::Point2f uv, float& z);

    int detectFeatures();
    int matchTwoFrames(Frame::Ptr frame_dst,
                       std::vector<int> &matchesRansac,
                       std::vector<cv::Point2f> &currFeatures,
                       bool showMatch);

    Frame::Ptr current_frame_ = nullptr;
    Frame::Ptr reference_frame_ = nullptr;

    Map::Ptr map_ = nullptr;

    // params
    int num_features_ = 200;
    int match_threshold_ = 50;

    // image
    int image_width = 1920;
    int image_height = 960;

    std::vector<Frame::Ptr> frames_;

    std::string match_save_dir_;

    double min_distance = 0.5;
    double max_distance = 60.0;
    
    // image pyramid level
    int maxLevel = 4;
    cv::Size winSize = cv::Size(31, 31);

    bool isInitialized;
};

#endif