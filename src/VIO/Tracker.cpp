#include <Tracker.h>

bool Tracker::addFrame(Frame::Ptr frame)
{

    if (map_->active_keyframes_.size() == 0)
    {
        current_frame_ = frame;
        init();
    }
    else
    {
        reference_frame_ = current_frame_;
        current_frame_ = frame;
        buildMap();
    }

    return true;
}

bool Tracker::init()
{
    int num_features = detectFeatures();
    LOG(INFO) << "# of features in first frame: " << num_features;
    insertKeyFrame();

    // Build initial map

    // new map point
    for (auto &feature : current_frame_->features_)
    {
        Vec2 p_c_last(feature->position_.pt.x, feature->position_.pt.y);
        Vec3 p_w_last = current_frame_->pixel2world(p_c_last, feature->depth_);
        auto new_map_point = MapPoint::createNewMappoint();
        new_map_point->setPos(p_w_last);
        new_map_point->rgb_ = feature->rgb_;
        new_map_point->addObservation(feature);
        new_map_point->id_frame_ = current_frame_->id_;

        feature->map_point_ = new_map_point;
        map_->insertMapPoint(new_map_point);
    }

    LOG(INFO) << "# of initial map points: " << map_->active_landmarks_.size();

    return true;
}

int Tracker::detectFeatures()
{
    // detect features
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    // detector_->detect(current_frame_->gray_, keypoints);

    double quality_level = 0.001;
    double min_distance_feature = 10;
    int block_size = 5;
    bool use_harris_detector = false;
    double k = 0.04;

    // Create GFTT detector
    cv::Ptr<cv::GFTTDetector> detector = cv::GFTTDetector::create(Config::Get<int>("num_features"), quality_level, min_distance_feature, block_size, use_harris_detector, k);

    detector->detect(current_frame_->gray_, keypoints);

    std::vector<cv::KeyPoint> keypoints_valid;

    int cnt_detected = 0;
    for (auto &kp : keypoints)
    {
        int u = (int)kp.pt.y;
        int v = (int)kp.pt.x;

        std::vector<double> rgb_{current_frame_->rgb_.at<cv::Vec3b>(kp.pt.y, kp.pt.x)[2] / 1.0,
                                 current_frame_->rgb_.at<cv::Vec3b>(kp.pt.y, kp.pt.x)[1] / 1.0,
                                 current_frame_->rgb_.at<cv::Vec3b>(kp.pt.y, kp.pt.x)[0] / 1.0};

        double z = 1.0;

        if (current_frame_->depthAvailable)
        {
            int depth_window_size = 11; // including center
            int x_left = v - (depth_window_size - 1) / 2;
            int x_right = v + (depth_window_size - 1) / 2;
            int y_upper = u - (depth_window_size - 1) / 2;
            int y_bottom = u + (depth_window_size - 1) / 2;

            bool isWindowValid = (x_left >= 0 && x_right <= image_width - 1 && y_upper >= 0 && y_bottom <= image_height - 1);

            if (!isWindowValid)
                continue;

            double z_aux = 0.0;
            double z_cnt = 0.0;

            for (int x = x_left; x < x_right + 1; x++)
            {
                for (int y = y_upper; y < y_bottom + 1; y++)
                {
                    double z_tmp = (double)current_frame_->depth_.at<float>(y, x);

                    if (z_tmp < min_distance || z_tmp > max_distance)
                        continue;

                    z_aux += z_tmp;
                    z_cnt += 1.0;
                }
            }

            if (z_cnt < 10)
                continue;

            z = z_aux / z_cnt;
        }

        current_frame_->features_.push_back(Feature::Ptr(new Feature(cnt_detected, current_frame_, kp, z, rgb_)));
        keypoints_valid.push_back(kp);

        cnt_detected++;
    }

    descriptor_->compute(current_frame_->gray_, keypoints_valid, current_frame_->descriptors_);

    return cnt_detected;
}

int Tracker::matchTwoFrames(Frame::Ptr frame_dst,
                            std::vector<cv::DMatch> &matches,
                            std::vector<cv::KeyPoint> &kps1,
                            std::vector<cv::KeyPoint> &kps2,
                            bool ransac,
                            bool showMatch)
{


    // calculate descriptor

    for (auto &feature : frame_dst->features_)
        kps1.emplace_back(feature->position_);

    for (auto &feature : current_frame_->features_)
        kps2.emplace_back(feature->position_);

    // why computer descriptor again?
    // descriptor_->compute(frame_dst->gray_, kps1, des_1);
    // descriptor_->compute(current_frame_->gray_, kps2, des_2);

    std::vector<std::vector<cv::DMatch>> matches_temp;

    // Need to improve
    if (Config::Get<std::string>("feature_type") == "ORB")
        matcher_ORB.knnMatch(frame_dst->descriptors_, current_frame_->descriptors_, matches_temp, 2);
    else if (Config::Get<std::string>("feature_type") == "SIFT")
        matcher_SIFT->knnMatch(frame_dst->descriptors_, current_frame_->descriptors_, matches_temp, 2);

    int max_query_id = -1;
    for (auto &match : matches_temp)
    {

        if (match.size() == 2)
        {
            if (match[0].distance < match[1].distance * knn_ratio_)
            {
                int query_id = match[0].queryIdx;
                if (query_id > max_query_id)
                    max_query_id = query_id;
                matches.emplace_back(match[0]);
            }
        }
    }

    if (max_query_id < 0)
        return 0;

    std::vector<bool> unique_match(max_query_id + 1);

    for (size_t i = 0; i < unique_match.size(); i++)
        unique_match[i] = false;

    if (ransac)
    {
        // filtering using ransac

        std::vector<cv::DMatch> matches_good;

        if (matches.size() > match_threshold_)
        {
            std::vector<cv::Point2f> kps_1_pt;
            std::vector<cv::Point2f> kps_2_pt;
            for (size_t i = 0; i < matches.size(); i++)
            {
                //-- Get the keypoints from the good matches
                kps_1_pt.emplace_back(kps1[matches[i].queryIdx].pt);
                kps_2_pt.emplace_back(kps2[matches[i].trainIdx].pt);
            }

            std::vector<int> mask;
            cv::Mat H = cv::findHomography(kps_1_pt, kps_2_pt, cv::RANSAC, 10, mask);

            for (size_t i = 0; i < mask.size(); i++)
            {
                if (mask[i] == 1)
                {
                    int query_id = matches[i].queryIdx;
                    if (!unique_match[query_id])
                        unique_match[query_id] = true;
                    else
                    {
                        LOG(INFO) << "FATAL ERROR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!";
                        continue;
                    }
                    matches_good.emplace_back(matches[i]);
                }
            }
        }
        matches = matches_good;
    }

    // Ransac filter

    if (showMatch)
    {
        cv::Mat img_match;

        cv::Mat img_ref = frame_dst->rgb_;
        cv::Mat img_cur = current_frame_->rgb_;
        cv::drawMatches(img_ref, kps1, img_cur, kps2, matches, img_match, cv::Scalar::all(-1), cv::Scalar::all(-1),
                        std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

        std::string title = "Match_" + std::to_string(frame_dst->id_) + "_" + std::to_string(current_frame_->id_);
        std::string save_dir = match_save_dir_ + "/" + title + ".jpg";
        cv::imwrite(save_dir, img_match);
        // cv::imshow(title, img_match);
        // cv::waitKey(0);
    }

    return (int)matches.size();
}

bool Tracker::buildMap()
{

    int num_features = detectFeatures(); // current frame feature detection
    Frame::Ptr kf = map_->current_keyframe_;

    std::vector<cv::DMatch> matches;
    std::vector<cv::KeyPoint> kps1, kps2;

    int num_matches = matchTwoFrames(kf, matches, kps1, kps2, true, true);

    std::vector<Eigen::Vector3d> pts1, pts2;
    // Eigen::Matrix3d K = camera_->K();

    for (size_t i = 0; i < matches.size(); i++)
    {
        auto m = matches[i];

        // link map point
        auto mp = kf->features_[m.queryIdx]->map_point_.lock();

        if (mp)
        {
            // existing map point
            current_frame_->features_[m.trainIdx]->map_point_ = kf->features_[m.queryIdx]->map_point_;

            // triangulation
            double z = current_frame_->getZvalue(mp->pos_);
            current_frame_->features_[m.trainIdx]->depth_ = z;

            mp->addObservation(current_frame_->features_[m.trainIdx]);
        }
        else
        {
            // new map point

            Vec2 p_c_last(kf->features_[m.queryIdx]->position_.pt.x, kf->features_[m.queryIdx]->position_.pt.y);
            Vec3 p_w_last = kf->pixel2world(p_c_last, kf->features_[m.queryIdx]->depth_);
            auto new_map_point = MapPoint::createNewMappoint();
            new_map_point->setPos(p_w_last);
            new_map_point->rgb_ = kf->features_[m.queryIdx]->rgb_;
            new_map_point->addObservation(kf->features_[m.queryIdx]);
            new_map_point->addObservation(current_frame_->features_[m.trainIdx]);
            new_map_point->id_frame_ = kf->id_;

            kf->features_[m.queryIdx]->map_point_ = new_map_point;
            current_frame_->features_[m.trainIdx]->map_point_ = new_map_point;

            map_->insertMapPoint(new_map_point);
        }
    }

    // Please.....
    // bool isOptimized = optimize();

    if (num_matches < match_threshold_)
        insertKeyFrame();

    if (!reference_frame_->is_keyframe_)
    {
        reference_frame_->rgb_ = cv::Mat();
        reference_frame_->gray_ = cv::Mat();
    }

    LOG(INFO) << "The number of keyframes: " << map_->keyframes_.size();
    LOG(INFO) << "The number of landmarks: " << map_->landmarks_.size();

    LOG(INFO) << "The number of active keyframes: " << map_->active_keyframes_.size();
    LOG(INFO) << "The number of active landmarks: " << map_->active_landmarks_.size();

    return true;
}

bool Tracker::optimize()
{
    Eigen::Matrix3d K = current_frame_->K_;
    int window_size = (int)(map_->active_keyframes_.size());

    Vec4 qvec_param[window_size + 1]; // active keyframes + current frame
    Vec3 tvec_param[window_size + 1];
    Vec3 lm_param[map_->active_landmarks_.size()];

    std::unordered_map<int, int> idx2idx;
    ceres::Problem problem;

    // Pose parameters
    int ii = 0;
    for (auto &kfs : map_->active_keyframes_)
    {
        idx2idx.insert(std::make_pair(kfs.second->id_, ii)); // frame_id -> id
        Eigen::Matrix4d pose_ = kfs.second->pose_;
        Eigen::Quaterniond q_eigen(pose_.block<3, 3>(0, 0));
        qvec_param[ii] = Vec4(q_eigen.w(), q_eigen.x(), q_eigen.y(), q_eigen.z());
        tvec_param[ii] = pose_.block<3, 1>(0, 3);
        problem.AddParameterBlock(qvec_param[ii].data(), 4);
        problem.AddParameterBlock(tvec_param[ii].data(), 3);
        LOG(INFO) << "PARAM BLOCK " << qvec_param[ii].data() << " ADDED";
        LOG(INFO) << "PARAM BLOCK " << tvec_param[ii].data() << " ADDED";
        ii += 1;
    }
    idx2idx.insert(std::make_pair(current_frame_->id_, ii));
    Eigen::Matrix4d pose_ = current_frame_->pose_;
    Eigen::Quaterniond q_eigen(pose_.block<3, 3>(0, 0));
    qvec_param[ii] = Vec4(q_eigen.w(), q_eigen.x(), q_eigen.y(), q_eigen.z());
    tvec_param[ii] = pose_.block<3, 1>(0, 3);
    problem.AddParameterBlock(qvec_param[ii].data(), 4);
    problem.AddParameterBlock(tvec_param[ii].data(), 3);
    LOG(INFO) << "PARAM BLOCK " << qvec_param[ii].data() << " ADDED";
    LOG(INFO) << "PARAM BLOCK " << tvec_param[ii].data() << " ADDED";

    // Landmark parameters (-->inverse depth)
    std::unordered_map<int, int> idx2idx_lm;
    int jj = 0;
    for (auto &lm : map_->active_landmarks_)
    {
        idx2idx_lm.insert(std::make_pair(lm.second->id_, jj));
        lm_param[jj] = lm.second->Pos();
        problem.AddParameterBlock(lm_param[jj].data(), 3);
        jj += 1;
    }
    // Add Reprojection factors
    for (auto &mp : map_->active_landmarks_)
    {
        int lm_id = idx2idx_lm.find(mp.second->id_)->second;
        for (auto &ob : mp.second->observations_)
        {
            auto feature = ob.lock();

            if (feature == nullptr)
                continue;

            auto item = idx2idx.find(feature->frame_.lock()->id_);
            if (item != idx2idx.end())
            {

                int frame_id = item->second;
                Eigen::Vector2d obs_src(ob.lock()->position_.pt.x, ob.lock()->position_.pt.y);

                ceres::CostFunction *cost_function = ProjectionFactorSimplePinholeConstantIntrinsic::Create(obs_src, K);
                problem.AddResidualBlock(cost_function,
                                         new ceres::CauchyLoss(0.5),
                                         qvec_param[frame_id].data(),
                                         tvec_param[frame_id].data(),
                                         lm_param[lm_id].data());
            }
        }
    }

    LOG(INFO) << "Make landmarks as constant";

    for (int i = 0; i < (int)map_->active_landmarks_.size(); i++)
    {
        problem.SetParameterBlockConstant(lm_param[i].data());
    }

    LOG(INFO) << "Make keyframe poses as constant";
    for (int i = 0; i < window_size; i++)
    {
        problem.SetParameterBlockConstant(qvec_param[i].data());
        problem.SetParameterBlockConstant(tvec_param[i].data());
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = false;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.num_threads = 8;
    options.max_num_iterations = 10000;
    // options.max_solver_time_in_seconds = 0.04;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << "\n";
    // Update params
    Eigen::Quaterniond Q;
    Vec3 Trans = tvec_param[window_size];

    Q.w() = qvec_param[window_size][0];
    Q.x() = qvec_param[window_size][1];
    Q.y() = qvec_param[window_size][2];
    Q.z() = qvec_param[window_size][3];

    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    pose.block<3, 3>(0, 0) = Q.normalized().toRotationMatrix();
    pose.block<3, 1>(0, 3) = Trans;
    LOG(INFO) << "OPTIMIZED POSE";
    std::cout << pose << std::endl;
    current_frame_->setPose(pose);

    // for (auto &mp : map_->active_landmarks_)
    // {
    //     int lm_id = idx2idx_lm.find(mp.second->id_)->second;
    //     mp.second->pos_ = lm_param[lm_id];
    // }

    return true;

    // // // // Write txt file

    // // // // cameras.txt
    // // // std::ofstream ofile("/home/seungwon/cameras.txt");
    // // // if (ofile.is_open())
    // // // {
    // // //     ofile << "# Camera list with one line of data per camera:\n";
    // // //     ofile << "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n";
    // // //     ofile << "# Number of cameras: 1\n";
    // // //     std::string str = "1 SIMPLE_RADIAL " +
    // // //                       std::to_string(image_width) + " " +
    // // //                       std::to_string(imnage_heighjt) + " " +
    // // //                       std::to_string(intrinsic_param[0][0]) + " " +
    // // //                       std::to_string(intrinsic_param[0][1]) + " " +
    // // //                       std::to_string(intrinsic_param[0][2]) + " 0.0\n";
    // // //     ofile << str;
    // // //     ofile.close();
    // // // }

    // // // // points3D.txt
    // // // std::ofstream ofile2("/home/seungwon/points3D.txt");
    // // // if (ofile2.is_open())
    // // // {
    // // //     ofile2 << "# 3D point list with one line of data per point:\n";
    // // //     ofile2 << "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n";
    // // //     ofile2 << "# Number of points: " + std::to_string((int)map_->landmarks_.size()) + ", mean track length: 0.0\n";

    // // //     for (auto &mp : map_->landmarks_)
    // // //     {
    // // //         int id = mp.second->id_;
    // // //         Vec3 position = mp.second->Pos();
    // // //         std::vector<double> rgb = mp.second->rgb_;
    // // //         std::string data = std::to_string(id) + " " +
    // // //                            std::to_string(position[0]) + " " +
    // // //                            std::to_string(position[1]) + " " +
    // // //                            std::to_string(position[2]) + " " +
    // // //                            std::to_string((int)rgb[0]) + " " +
    // // //                            std::to_string((int)rgb[1]) + " " +
    // // //                            std::to_string((int)rgb[2]) + " 0";

    // // //         for (auto &ob_ptr : mp.second->observations_)
    // // //         {

    // // //             auto ob = ob_ptr.lock();

    // // //             if (ob)
    // // //                 data += (" " + std::to_string(ob->frame_.lock()->id_) + " " + std::to_string(ob->id_));
    // // //         }

    // // //         data += "\n";

    // // //         ofile2 << data;
    // // //     }
    // // //     ofile2.close();
    // // // }

    // // // // images.txt
    // // // std::ofstream ofile3("/home/seungwon/images.txt");

    // // // int cnt = 1;
    // // // if (ofile3.is_open())
    // // // {
    // // //     ofile3 << "# Image list with two lines of data per image:\n";
    // // //     ofile3 << "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n";
    // // //     ofile3 << "#   POINTS2D[] as (X, Y, POINT3D_ID)\n";
    // // //     ofile3 << ("# Number of images: " + std::to_string(map_->keyframes_.size()) + ", mean observations per image: 2000" + "\n");

    // // //     for (auto &kf_map : map_->keyframes_)
    // // //     {
    // // //         std::string data;
    // // //         std::string kf_id = std::to_string(kf_map.second->id_);
    // // //         std::string kf_name = kf_id + ".jpg";

    // // //         SE3 pose_ = kf_map.second->Pose();

    // // //         LOG(INFO) << pose_.translation();
    // // //         Eigen::Quaterniond q_eigen(pose_.rotationMatrix());
    // // //         q_eigen.normalize();
    // // //         Vec4 q = Vec4(q_eigen.w(), q_eigen.x(), q_eigen.y(), q_eigen.z());
    // // //         Vec3 t = pose_.translation();
    // // //         std::string q_str = std::to_string(q[0]) + " " + std::to_string(q[1]) + " " + std::to_string(q[2]) + " " + std::to_string(q[3]) + " ";
    // // //         std::string t_str = std::to_string(t[0]) + " " + std::to_string(t[1]) + " " + std::to_string(t[2]) + " ";
    // // //         data = std::to_string(cnt) + " " + q_str + t_str + std::to_string(1) + " " + kf_name + "\n";
    // // //         ofile3 << data;

    // // //         auto kf = kf_map.second;
    // // //         std::string data2;
    // // //         int cc = 0;
    // // //         for (auto feature_ptr : kf->features_)
    // // //         {
    // // //             if (cc > 0)
    // // //                 data2 += " ";
    // // //             std::string x_str = std::to_string(feature_ptr->position_.pt.x);
    // // //             std::string y_str = std::to_string(feature_ptr->position_.pt.y);
    // // //             auto mp = feature_ptr->map_point_.lock();
    // // //             std::string temp;
    // // //             if (mp)
    // // //                 temp = std::to_string(mp->id_);
    // // //             else
    // // //                 temp = "-1";

    // // //             data2 += (x_str + " " + y_str + " " + temp);
    // // //             cc += 1;
    // // //         }

    // // //         data2 += "\n";
    // // //         ofile3 << data2;
    // // //         cnt += 1;
    // // //     }
    // // //     ofile3.close();
    // // // }
}

bool Tracker::insertKeyFrame()
{
    // LOG(INFO) << "Try Set Keyframe";
    current_frame_->setKeyFrame();
    // LOG(INFO) << "Try Insert Keyframe";
    map_->insertKeyFrame(current_frame_);
    // LOG(INFO) << "Done";

    // setObservationsForKeyFrame();

    return true;
}

void Tracker::setObservationsForKeyFrame()
{
    for (auto &feat : current_frame_->features_)
    {
        auto mp = feat->map_point_.lock();
        if (mp)
            mp->addObservation(feat);
    }
}
