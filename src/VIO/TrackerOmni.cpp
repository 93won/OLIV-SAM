#include <TrackerOmni.h>
#include <G2OFactors.h>

bool TrackerOmni::addFrame(Frame::Ptr frame)
{

    if (!isInitialized)
    {
        current_frame_ = frame;
        if (init())
            isInitialized = true;
    }
    else
    {
        reference_frame_ = current_frame_;
        current_frame_ = frame;
        Track();
    }

    return true;
}

bool TrackerOmni::associateWithDepth(const cv::Point2f uv, float &z)
{

    int depth_window_size = 11; // including center
    int x_left = (int)uv.x - (depth_window_size - 1) / 2;
    int x_right = (int)uv.x + (depth_window_size - 1) / 2;
    int y_upper = (int)uv.y - (depth_window_size - 1) / 2;
    int y_bottom = (int)uv.y + (depth_window_size - 1) / 2;

    bool isWindowValid = (x_left >= 0 && x_right <= image_width - 1 && y_upper >= 0 && y_bottom <= image_height - 1);

    if (!isWindowValid)
        return false;

    float z_aux = 0.0;
    float z_cnt = 0.0;

    for (int x = x_left; x < x_right + 1; x++)
    {
        for (int y = y_upper; y < y_bottom + 1; y++)
        {
            float z_tmp = current_frame_->depth_.at<float>(y, x);

            if (z_tmp < min_distance || z_tmp > max_distance)
                continue;

            z_aux += z_tmp;
            z_cnt += 1.0;
        }
    }

    if (z_cnt < 10)
    {
        return false;
    }

    z = z_aux / z_cnt;

    return true;
}

bool TrackerOmni::init()
{

    if (!current_frame_->depthAvailable)
        return false;

    // // Step 1: Detect initial features
    double qualityLevel = 0.01;
    double minDistance = 30;
    int blockSize = 3, gradientSize = 3;
    bool useHarrisDetector = false;
    double k = 0.04;
    bool imgInit = false;

    // Create mask with same size as image, initially all zeros (black)
    cv::Mat maskGlobal = cv::Mat::zeros(cv::Size(1920, 960), CV_8U);

    // Set middle half of the mask to white (255)
    maskGlobal.rowRange(960 / 4, 960 * 3 / 4).setTo(cv::Scalar(255));

    cv::equalizeHist(current_frame_->gray_, current_frame_->gray_);

    cv::TermCriteria termcrit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03);
    cv::Size subPixWinSize(10, 10);
    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(current_frame_->gray_, corners, 500, qualityLevel, minDistance, maskGlobal, blockSize, gradientSize, useHarrisDetector, k);
    cv::cornerSubPix(current_frame_->gray_, corners, subPixWinSize, cv::Size(-1, -1), termcrit);

    std::vector<cv::KeyPoint> keypoints;

    std::vector<double> rgb_ = {1.0, 0.0, 0.0};

    int cnt_detected = 0;

    for (auto &corner : corners)
    {
        cv::KeyPoint kp(corner.x, corner.y, 1.0);
        float z = 1.0;
        bool isAssociated = false;
        if (associateWithDepth(corner, z))
        {
            current_frame_->features_.push_back(Feature::Ptr(new Feature(cnt_detected++, current_frame_, kp, (double)z, isAssociated)));
        }
    }

    // // Step 2: Insert new keyframe
    insertKeyFrame();


    cv::buildOpticalFlowPyramid(map_->current_keyframe_->gray_, map_->current_keyframe_->imgPyramid_, winSize, maxLevel);

    // // Step 3: Build initial map

    for (auto &feature : current_frame_->features_)
    {
        Vec2 p_c_last(feature->position_.pt.x, feature->position_.pt.y);
        Vec3 p_w_last = current_frame_->pixel2world(p_c_last, feature->depth_); // TODO: change it to 360 version
        auto new_map_point = MapPoint::createNewMappoint();
        new_map_point->setPos(p_w_last);
        new_map_point->rgb_ = feature->rgb_;
        new_map_point->addObservation(feature);
        new_map_point->id_frame_ = current_frame_->id_;

        std::cout << "CHECK POSITION: " << p_w_last.transpose() << " / " << feature->depth_ << std::endl;

        feature->map_point_ = new_map_point;
        map_->insertMapPoint(new_map_point);
    }

    LOG(INFO) << "# of initial map points: " << map_->active_landmarks_.size();

    return true;
}

bool TrackerOmni::Track()
{
    bool debug = false;

    if (debug)
        LOG(INFO) << "[DEBUG] Tracking start";
    // // Step 1: Optical Flow
    double qualityLevel = 0.01;
    double minDistance = 30;
    int blockSize = 3, gradientSize = 3;
    bool useHarrisDetector = false;
    double k = 0.04;
    bool imgInit = false;

    // Create mask with same size as image, initially all zeros (black)
    cv::Mat maskGlobal = cv::Mat::zeros(cv::Size(1920, 960), CV_8U);

    // Set middle half of the mask to white (255)
    maskGlobal.rowRange(960 / 4, 960 * 3 / 4).setTo(cv::Scalar(255));

    cv::Mat mask = maskGlobal.clone();

    // Update mask to exclude points that have been detected

    std::vector<cv::Point2f> prevPts, currPts;

    if (debug)
        LOG(INFO) << "[DEBUG] Try make mask";

    for (auto &feature : reference_frame_->features_)
        prevPts.emplace_back(cv::Point2f(feature->position_.pt.x, feature->position_.pt.y));

    for (const auto &pt : prevPts)
        cv::circle(mask, pt, 50, cv::Scalar(0), -1);

    // Detect New corner points
    std::vector<cv::Point2f> newCorners;

    cv::TermCriteria termcrit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03);
    cv::Size subPixWinSize(10, 10);
    cv::equalizeHist(current_frame_->gray_, current_frame_->gray_);

    std::vector<uchar> status;
    std::vector<uchar> status_ransac;
    std::vector<float> err;

    std::vector<cv::Mat> prevPyramid, currPyramid;

    if (debug)
        LOG(INFO) << "[DEBUG] Try build optical flow pyramid";

    cv::buildOpticalFlowPyramid(current_frame_->gray_, current_frame_->imgPyramid_, winSize, maxLevel);

    if (debug)
        LOG(INFO) << "[DEBUG] Try track features using optical flow";

    cv::calcOpticalFlowPyrLK(reference_frame_->imgPyramid_, current_frame_->imgPyramid_, prevPts, currPts, status, err, winSize, maxLevel);

    if (debug)
        LOG(INFO) << "[DEBUG] Try using ransac for outlier removal";

    // Only keep points for which optical flow was actually found
    std::vector<cv::Point2f> prevGood, currGood;
    std::vector<int> matches, matchesRobust;
    for (size_t k = 0; k < currPts.size(); k++)
    {
        if (status[k])
        {
            matches.push_back(k);
            prevGood.push_back(prevPts[k]);
            currGood.push_back(currPts[k]);
        }
    }

    std::vector<cv::Point2f> prevBetter, currBetter;

    // cv::findFundamentalMat(prevGood, currGood, cv::FM_RANSAC, 1.0, 0.99, status_ransac);

    for (size_t k = 0; k < currGood.size(); k++)
    {
        // if (status_ransac[k])
        {
            matchesRobust.push_back(matches[k]);
            prevBetter.push_back(prevGood[k]);
            currBetter.push_back(currGood[k]);
        }
    }

    if (debug)
        LOG(INFO) << "[DEBUG] Try add tracked features to current frame and drawing tracking result";
    size_t numTrackedPoints = currBetter.size();

    std::vector<double> rgb_ = {1.0, 0.0, 0.0};

    int cnt_detected = 0;

    float maxObservation = 30.0;

    for (size_t k = 0; k < matchesRobust.size(); k++)
    {
        auto corner = currBetter[k];
        int idxFeatureKf = matchesRobust[matches[k]];

        cv::KeyPoint kp(corner.x, corner.y, 1.0);

        float z = 1.0;
        bool isAssociated = false;

        if (current_frame_->depthAvailable)
            isAssociated = associateWithDepth(corner, z);

        current_frame_->features_.push_back(Feature::Ptr(new Feature(cnt_detected, current_frame_, kp, (double)z, isAssociated)));

        // // Visualization
        // auto mp = reference_frame_->features_[matchesRobust[k]]->map_point_.lock();
        // if (mp)
        // {
        //     int numObs = int(mp->observations_.size());
        //     int colorRed = (int)(((float)numObs)/maxObservation*255.0);
        //     cv::line(current_frame_->rgb_, prevBetter[cnt_detected], corner, cv::Scalar(0, 255, 0), 2, 4, 0);
        //     cv::circle(current_frame_->rgb_, corner, 3, cv::Scalar(255-colorRed, 0, colorRed), -1, 4, 0);
        // }

        cnt_detected += 1;
    }

    // for (auto &corner : currBetter)
    // {
    //     cv::KeyPoint kp(corner.x, corner.y, 1.0);

    //     float z = 1.0;
    //     bool isAssociated = false;
    //     if (current_frame_->depthAvailable)
    //         isAssociated = associateWithDepth(corner, z);

    //     cv::line(current_frame_->rgb_, prevBetter[cnt_detected], corner, cv::Scalar(0, 255, 0), 2, 4, 0);
    //     cv::circle(current_frame_->rgb_, corner, 3, cv::Scalar(0, 0, 255), -1, 4, 0);

    //     current_frame_->features_.push_back(Feature::Ptr(new Feature(cnt_detected++, current_frame_, kp, (double)z, isAssociated)));
    // }

    if (debug)
        LOG(INFO) << "[DEBUG] Try make link between feature and map point";

    // matchesRobust: vector of matched feature indices of reference keyframe

    for (size_t idx = 0; idx < matchesRobust.size(); idx++)
    {
        cv::Point2f uvFeature(current_frame_->features_[idx]->position_.pt.x, current_frame_->features_[idx]->position_.pt.y);

        auto mp = reference_frame_->features_[matchesRobust[idx]]->map_point_.lock();
        if (mp)
        {
            // if map point exist, just link!
            current_frame_->features_[idx]->map_point_ = reference_frame_->features_[matchesRobust[idx]]->map_point_;
            mp->addObservation(current_frame_->features_[idx]);

            int numObs = int(mp->observations_.size());

            if ((float)numObs > maxObservation)
                numObs = (int)maxObservation;

            int colorRed = (int)(((float)numObs) / maxObservation * 255.0);

            cv::circle(current_frame_->rgb_, uvFeature, 3, cv::Scalar(255 - colorRed, 255 - colorRed, colorRed), -1, 4, 0);
        }
        else
        {
            // if feature tracked but no map point related, add new map point!

            if (!reference_frame_->features_[matchesRobust[idx]]->isAssociated_)
                continue;

            Vec2 uv_feature_kf(reference_frame_->features_[matchesRobust[idx]]->position_.pt.x,
                               reference_frame_->features_[matchesRobust[idx]]->position_.pt.y);

            Vec3 xyz_feature_kf_world = reference_frame_->pixel2world(uv_feature_kf,
                                                                      reference_frame_->features_[matchesRobust[idx]]->depth_);

            auto newMapPoint = MapPoint::createNewMappoint();
            newMapPoint->setPos(xyz_feature_kf_world);
            newMapPoint->addObservation(reference_frame_->features_[matchesRobust[idx]]);
            newMapPoint->addObservation(current_frame_->features_[idx]);
            newMapPoint->id_frame_ = reference_frame_->id_;

            reference_frame_->features_[matchesRobust[idx]]->map_point_ = newMapPoint;
            current_frame_->features_[idx]->map_point_ = newMapPoint;

            map_->insertMapPoint(newMapPoint);

            int numObs = int(newMapPoint->observations_.size());

            if ((float)numObs > maxObservation)
                numObs = (int)maxObservation;

            int colorRed = (int)(((float)numObs) / maxObservation * 255.0);
            cv::circle(current_frame_->rgb_, uvFeature, 3, cv::Scalar(255 - colorRed, 255 - colorRed, colorRed), -1, 4, 0);
        }
    }

    bool needNewKeyFrame = (numTrackedPoints < 200 && current_frame_->depthAvailable);

    LOG(INFO) << "The number of tracked points: " << numTrackedPoints;

    if (needNewKeyFrame)
    {
        cv::goodFeaturesToTrack(current_frame_->gray_, newCorners, 500, qualityLevel, minDistance, mask, blockSize, gradientSize, useHarrisDetector, k);

        if (!newCorners.empty())
        {
            cv::cornerSubPix(current_frame_->gray_, newCorners, subPixWinSize, cv::Size(-1, -1), termcrit);
            // Append new corners to existing points
            for (auto &corner : newCorners)
            {
                // cv::KeyPoint kp(corner.x, corner.y, 1.0);
                // current_frame_->features_.push_back(Feature::Ptr(new Feature(cnt_detected++, current_frame_, kp, 1.0, rgb_)));

                cv::KeyPoint kp(corner.x, corner.y, 1.0);

                float z = 1.0;
                bool isAssociated = false;
                if (current_frame_->depthAvailable)
                    isAssociated = associateWithDepth(corner, z);

                current_frame_->features_.push_back(Feature::Ptr(new Feature(cnt_detected++, current_frame_, kp, (double)z, isAssociated)));
            }
            // prevPts.insert(prevPts.end(), newCorners.begin(), newCorners.end());
        }
        insertKeyFrame();
    }

    cv::imshow("Optical Flow", current_frame_->rgb_);
    cv::waitKey(1);

    if (!reference_frame_->is_keyframe_)
    {
        reference_frame_->rgb_ = cv::Mat();
        reference_frame_->gray_ = cv::Mat();
    }

    LOG(INFO) << "The number of keyframes: " << map_->keyframes_.size();
    LOG(INFO) << "The number of landmarks: " << map_->landmarks_.size();

    // LOG(INFO) << "The number of active keyframes: " << map_->active_keyframes_.size();
    // LOG(INFO) << "The number of active landmarks: " << map_->active_landmarks_.size();
}

// int TrackerOmni::detectFeatures()
// {

//     // Detect new feature points
//     double qualityLevel = 0.1;
//     double minDistance = 30;
//     int blockSize = 3, gradientSize = 3;
//     bool useHarrisDetector = false;
//     double k = 0.04;
//     bool imgInit = false;

//     // Create mask with same size as image, initially all zeros (black)
//     cv::Mat maskGlobal = cv::Mat::zeros(cv::Size(1920, 960), CV_8U);

//     // Set middle half of the mask to white (255)
//     maskGlobal.rowRange(960 / 4, 960 * 3 / 4).setTo(cv::Scalar(255));

//     cv::Mat mask = maskGlobal.clone();

//     // Update mask to exclude points that have been detected

//     std::vector<cv::Point2f> prevPts;

//     for (auto &feature : current_frame_->features_)
//         prevPts.emplace_back(cv::Point2f(feature->position_.pt.x, feature->position_.pt.y));

//     for (const auto &pt : prevPts)
//         cv::circle(mask, pt, 25, cv::Scalar(0), -1);

//     cv::equalizeHist(current_frame_->gray_, current_frame_->gray_);

//     cv::TermCriteria termcrit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03);
//     cv::Size subPixWinSize(10, 10);
//     std::vector<cv::Point2f> corners;
//     cv::goodFeaturesToTrack(current_frame_->gray_, corners, 500, qualityLevel, minDistance, mask, blockSize, gradientSize, useHarrisDetector, k);
//     cv::cornerSubPix(current_frame_->gray_, corners, subPixWinSize, cv::Size(-1, -1), termcrit);

//     std::vector<cv::KeyPoint> keypoints;

//     for (auto &corner : corners)
//         keypoints.push_back(cv::KeyPoint(corner.x, corner.y, 1.0));

//     int cnt_detected = 0;
//     for (auto &kp : keypoints)
//     {
//         int u = (int)kp.pt.y;
//         int v = (int)kp.pt.x;

//         std::vector<double> rgb_{current_frame_->rgb_.at<cv::Vec3b>(kp.pt.y, kp.pt.x)[2] / 1.0,
//                                  current_frame_->rgb_.at<cv::Vec3b>(kp.pt.y, kp.pt.x)[1] / 1.0,
//                                  current_frame_->rgb_.at<cv::Vec3b>(kp.pt.y, kp.pt.x)[0] / 1.0};

//         double z = 1.0;

//         if (current_frame_->depthAvailable)
//         {
//             int depth_window_size = 11; // including center
//             int x_left = v - (depth_window_size - 1) / 2;
//             int x_right = v + (depth_window_size - 1) / 2;
//             int y_upper = u - (depth_window_size - 1) / 2;
//             int y_bottom = u + (depth_window_size - 1) / 2;

//             bool isWindowValid = (x_left >= 0 && x_right <= image_width - 1 && y_upper >= 0 && y_bottom <= image_height - 1);

//             if (!isWindowValid)
//                 continue;

//             double z_aux = 0.0;
//             double z_cnt = 0.0;

//             for (int x = x_left; x < x_right + 1; x++)
//             {
//                 for (int y = y_upper; y < y_bottom + 1; y++)
//                 {
//                     double z_tmp = (double)current_frame_->depth_.at<float>(y, x);

//                     if (z_tmp < min_distance || z_tmp > max_distance)
//                         continue;

//                     z_aux += z_tmp;
//                     z_cnt += 1.0;
//                 }
//             }

//             if (z_cnt < 10)
//                 continue;

//             z = z_aux / z_cnt;
//         }

//         current_frame_->features_.push_back(Feature::Ptr(new Feature(cnt_detected, current_frame_, kp, z, rgb_)));

//         cnt_detected++;
//     }

//     return cnt_detected;
// }

// int TrackerOmni::matchTwoFrames(Frame::Ptr frame_dst,
//                                 std::vector<int> &matchesRansac,
//                                 std::vector<cv::Point2f> &currFeatures,
//                                 bool showMatch)
// {

//     std::vector<cv::Mat> prevPyramid, currPyramid;
//     cv::Size winSize(31, 31);
//     int maxLevel = 4;

//     cv::buildOpticalFlowPyramid(frame_dst->gray_, prevPyramid, winSize, maxLevel);
//     cv::buildOpticalFlowPyramid(current_frame_->gray_, currPyramid, winSize, maxLevel);

//     std::vector<cv::Point2f> prevPts, currPts;

//     for (auto &feature : frame_dst->features_)
//         prevPts.emplace_back(cv::Point2f(feature->position_.pt.x, feature->position_.pt.y));

//     // for (auto &feature : current_frame_->features_)
//     //     currPts.emplace_back(cv::Point2f(feature->position_.pt.x, feature->position_.pt.y));

//     std::vector<uchar> status;
//     std::vector<uchar> status_ransac;
//     std::vector<float> err;

//     cv::calcOpticalFlowPyrLK(prevPyramid, currPyramid, prevPts, currPts, status, err, winSize, maxLevel);

//     // Only keep points for which optical flow was actually found
//     std::vector<cv::Point2f> prevGood, currGood;
//     std::vector<int> matches;
//     for (size_t k = 0; k < currPts.size(); k++)
//     {
//         if (status[k])
//         {
//             matches.push_back(k);
//             prevGood.push_back(prevPts[k]);
//             currGood.push_back(currPts[k]);
//             // cv::line(currFrame, prevPts[k], currPts[k], cv::Scalar(0, 255, 0), 2, 4, 0);
//             // cv::circle(currFrame, currPts[k], 3, cv::Scalar(0, 0, 255), -1, 4, 0);
//         }
//     }

//     std::vector<cv::Point2f> prevBetter;

//     cv::findFundamentalMat(prevGood, currGood, cv::FM_RANSAC, 1.0, 0.99, status_ransac);

//     for (size_t k = 0; k < currGood.size(); k++)
//     {
//         if (status_ransac[k])
//         {
//             matchesRansac.push_back(matches[k]);
//             currFeatures.push_back(currGood[k]);

//             if (showMatch)
//             {
//                 cv::line(current_frame_->rgb_, prevGood[k], currGood[k], cv::Scalar(0, 255, 0), 2, 4, 0);
//                 cv::circle(current_frame_->rgb_, currGood[k], 3, cv::Scalar(0, 0, 255), -1, 4, 0);
//             }
//         }
//     }

//     if (showMatch)
//     {
//         cv::imshow("OpticalFlow", current_frame_->rgb_);
//         cv::waitKey(1);
//     }

//     return (int)matches.size();
// }

// bool TrackerOmni::buildMap()
// {

//     Frame::Ptr kf = map_->current_keyframe_;
//     std::vector<int> matchesRansac;
//     std::vector<cv::Point2f> currFeatures;

//     int num_matches = matchTwoFrames(kf, matchesRansac, currFeatures, true);

//     LOG(INFO) << "### NUM MATCHES: " << num_matches;

//     std::vector<Eigen::Vector3d> pts1, pts2;
//     // Eigen::Matrix3d K = camera_->K();

//     for (size_t i = 0; i < matchesRansac.size(); i++)
//     {

//         int matchId = matchesRansac[i];

//         // TODO : associate depth

//         std::vector<double> rgb_{current_frame_->rgb_.at<cv::Vec3b>(currFeatures[i].y, currFeatures[i].x)[2] / 1.0,
//                                  current_frame_->rgb_.at<cv::Vec3b>(currFeatures[i].y, currFeatures[i].x)[1] / 1.0,
//                                  current_frame_->rgb_.at<cv::Vec3b>(currFeatures[i].y, currFeatures[i].x)[0] / 1.0};

//         current_frame_->features_.push_back(Feature::Ptr(new Feature((int)(current_frame_->features_.size() - 1),
//                                                                      current_frame_,
//                                                                      cv::KeyPoint(currFeatures[i].x, currFeatures[i].y, 1),
//                                                                      1.0,
//                                                                      rgb_)));

//         // link map point
//         auto mp = kf->features_[matchId]->map_point_.lock();

//         if (mp)
//         {
//             // existing map point
//             current_frame_->features_[i]->map_point_ = kf->features_[matchId]->map_point_;

//             // triangulation
//             // double z = current_frame_->getZvalue(mp->pos_);
//             current_frame_->features_[i]->depth_ = 1.0;

//             mp->addObservation(current_frame_->features_[i]);
//         }
//         // else
//         // {
//         //     // new map point

//         //     Vec2 p_c_last(kf->features_[matchId]->position_.pt.x, kf->features_[matchId]->position_.pt.y);
//         //     Vec3 p_w_last = kf->pixel2world(p_c_last, kf->features_[m.queryIdx]->depth_);
//         //     auto new_map_point = MapPoint::createNewMappoint();
//         //     new_map_point->setPos(p_w_last);
//         //     new_map_point->rgb_ = kf->features_[m.queryIdx]->rgb_;
//         //     new_map_point->addObservation(kf->features_[m.queryIdx]);
//         //     new_map_point->addObservation(current_frame_->features_[m.trainIdx]);
//         //     new_map_point->id_frame_ = kf->id_;

//         //     kf->features_[m.queryIdx]->map_point_ = new_map_point;
//         //     current_frame_->features_[m.trainIdx]->map_point_ = new_map_point;

//         //     map_->insertMapPoint(new_map_point);
//         // }
//     }

//     // if (num_matches < match_threshold_)
//     insertKeyFrame();

//     // LOG(INFO) << "The number of keyframes: " << map_->keyframes_.size();
//     // LOG(INFO) << "The number of landmarks: " << map_->landmarks_.size();

//     // LOG(INFO) << "The number of active keyframes: " << map_->active_keyframes_.size();
//     // LOG(INFO) << "The number of active landmarks: " << map_->active_landmarks_.size();

//     int num_features = detectFeatures(); // current frame feature detection

//     return true;
// }

bool TrackerOmni::insertKeyFrame()
{
    map_->current_keyframe_ = current_frame_;
    current_frame_->setKeyFrame();
    map_->insertKeyFrame(current_frame_);

    return true;
}