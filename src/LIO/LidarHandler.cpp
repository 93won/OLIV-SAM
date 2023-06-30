#include <LidarHandler.h>
#include <ctime>
// Objective Functions

void estimateNormal(const pcl::PointCloud<PointType>::Ptr &src,
                    const pcl::PointCloud<pcl::Normal>::Ptr &cloud_normal,
                    std::vector<bool> &planeValid)
{

    pcl::search::KdTree<PointType>::Ptr tree(new pcl::search::KdTree<PointType>());
    tree->setInputCloud(src);

    int num_pc = src->points.size();

    pcl::NormalEstimation<PointType, pcl::Normal> ne;
    ne.setInputCloud(src);
    ne.setSearchMethod(tree);
    ne.setKSearch(5);
    ne.compute(*cloud_normal);

    // #pragma omp parallel for num_threads(4)
    for (int i = 0; i < num_pc; ++i)
    {
        if (isnan(cloud_normal->points[i].normal_x))
            planeValid[i] = false;
        else
            planeValid[i] = true;
    }
}

Eigen::Matrix<double, 3, 3> skew(Eigen::Matrix<double, 3, 1> mat_in)
{
    Eigen::Matrix<double, 3, 3> skew_mat;
    skew_mat.setZero();
    skew_mat(0, 1) = -mat_in(2);
    skew_mat(0, 2) = mat_in(1);
    skew_mat(1, 2) = -mat_in(0);
    skew_mat(1, 0) = mat_in(2);
    skew_mat(2, 0) = -mat_in(1);
    skew_mat(2, 1) = mat_in(0);
    return skew_mat;
}

Eigen::Matrix<double, 3, 3> axisAngleToR(const Eigen::Vector3d &axisAngle)
{
    Eigen::Matrix3d X = Eigen::Matrix3d::Identity();

    if (axisAngle[0] == 0 && axisAngle[1] == 0 && axisAngle[2] == 0)
        X.setZero();
    else
        X = skew(axisAngle);

    double theta = X.determinant() + 1e-7;
    double s = sin(theta);
    double s_half = sin(theta / 2);

    Eigen::Matrix3d I33 = Eigen::Matrix3d::Identity();

    Eigen::Matrix3d R = I33 + (s / theta) * X + (2 * (s_half * s_half) / (theta * theta)) * (X * X);

    return R;
}

SurfSymmAnalyticCostFunction::SurfSymmAnalyticCostFunction(Eigen::Vector3d p_, Eigen::Vector3d np_,
                                                           Eigen::Vector3d q_, Eigen::Vector3d nq_)
    : p(p_), np(np_), q(q_), nq(nq_) {}

bool SurfSymmAnalyticCostFunction::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{

    // cost = (Rp+t - q)*(np+nq)
    // q and t is relative rotation and translation
    Eigen::Vector3d delta_R_axis_angle(parameters[0] + 3);
    Eigen::Matrix3d delta_R = axisAngleToR(delta_R_axis_angle);
    Eigen::Vector3d delta_t(parameters[0]);

    Eigen::Matrix3d skew_p = skew(delta_R * p + delta_t);
    Eigen::Vector3d A = (delta_R * p + delta_t - q);
    // Eigen::Vector3d B = np + nq; // if nq only, then it is point-to-plane
    Eigen::Vector3d B = np + nq; // if nq only, then it is point-to-plane

    residuals[0] = A.dot(B);

    if (jacobians != NULL)
    {
        if (jacobians[0] != NULL)
        {

            Eigen::Matrix<double, 3, 6> dT_by_se3;
            dT_by_se3.block<3, 3>(0, 3) = -skew_p;
            (dT_by_se3.block<3, 3>(0, 0)).setIdentity();

            Eigen::Map<Eigen::Matrix<double, 1, 6, Eigen::RowMajor>> J_se3(jacobians[0]);
            J_se3.setZero();
            J_se3.block<1, 6>(0, 0) = B.transpose() * dT_by_se3;
        }
    }

    return true;
}

bool PoseSE3SymmParameterization::Plus(const double *x, const double *delta, double *x_plus_delta) const
{

    Eigen::Map<const Eigen::Vector3d> trans(x);
    Eigen::Map<const Eigen::Vector3d> axisAngle(x + 3);

    Eigen::Map<const Eigen::Vector3d> deltaTrans(delta);
    Eigen::Map<const Eigen::Vector3d> deltaAxisAngle(delta + 3);

    Eigen::Map<Eigen::Vector3d> axis_angle_plus(x_plus_delta + 3);
    Eigen::Map<Eigen::Vector3d> trans_plus(x_plus_delta);

    axis_angle_plus = deltaAxisAngle + axisAngle;
    trans_plus = deltaTrans + trans;

    return true;
}

bool PoseSE3SymmParameterization::ComputeJacobian(const double *x, double *jacobian) const
{
    Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> j(jacobian);
    (j.topRows(6)).setIdentity();

    return true;
}

void convertPointType(const cloudXYZIT::Ptr &cloud_in, const cloudXYZ::Ptr &cloud_out)
{
    for (auto &p : cloud_in->points)
        cloud_out->push_back(pcl::PointXYZ(p.x, p.y, p.z));
}

void voxelDownSample(pcl::PointCloud<PointType>::Ptr &cloud_in, const double voxel_size = 0.1)
{
    pcl::PointCloud<PointType>::Ptr cloud_filtered(new pcl::PointCloud<PointType>);
    pcl::VoxelGrid<PointType> sor;

    sor.setLeafSize(voxel_size, voxel_size, voxel_size);

    sor.setInputCloud(cloud_in);
    sor.filter(*cloud_filtered);

    cloud_in = cloud_filtered;
}


void voxelDownSample(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud_in, const double voxel_size)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::VoxelGrid<pcl::PointXYZRGB> sor;

    sor.setLeafSize(voxel_size, voxel_size, voxel_size);

    sor.setInputCloud(cloud_in);
    sor.filter(*cloud_filtered);

    cloud_in = cloud_filtered;
}


void LidarHandler::updateOdometry(const pcl::PointCloud<PointType>::Ptr &cloud, const bool fix)
{

    pcl::PointCloud<PointType>::Ptr cloud_w(new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr cloud_surf(new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr cloud_surf_ds(new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr cloud_surf_ds_deskew(new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr cloud_surf_ds_w(new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr cloud_surf_map_local(new pcl::PointCloud<PointType>);


    // Step 1 - surface feature extraction
    featureExtraction(cloud, cloud_surf);

    // Step 2 - voxel downsampling
    voxelDownSample(cloud_surf, voxel_size);

    cloud_surf_ds = cloud_surf;

    // cloudXYZ::Ptr cloud_surf_xyz(new cloudXYZ);
    // cloudXYZ::Ptr cloud_surf_xyz_w(new cloudXYZ);
    // cloudXYZ::Ptr cloud_aligned(new cloudXYZ);
    // cloudXYZ::Ptr cloud_surf_map_local(new cloudXYZ);
    // cloudXYZ::Ptr cloud_surf_map_filtered(new cloudXYZ);
    // cloudXYZ::Ptr cloud_w_xyz(new cloudXYZ);

    // Step 3 - iterative closest point and update map

    // Initailization
    if (!lidarInit)
    {
        lidarInit = true;

        pcl::transformPointCloud(*cloud_surf_ds, *cloud_surf_ds_w, T_wl);
        cloud_surf_ref = cloud_surf_ds;
        cloud_surf_map = cloud_surf_ds_w;

        return;
    }

    // // Iterative closest point between reference cloud and current cloud
    // Eigen::Matrix4d T_curr_ref = iterativeClosestPoint(cloud_surf_ds, cloud_surf_ref); // transfrom from reference frame tot current frame
    // T_wl *= T_curr_ref;

    T_wl *= T_odom_ref;

    // Iterative closest point between map cloud and current cloud
    pcl::transformPointCloud(*cloud_surf_map, *cloud_surf_map_local, T_wl.inverse().cast<float>());
    Eigen::Matrix4d T_odom = iterativeClosestPoint(cloud_surf_ds, cloud_surf_map_local);
    T_wl *= T_odom;

    T_odom_ref = T_odom;

    // Transform local surface features to world coordinate system and merge them to map cloud
    pcl::transformPointCloud(*cloud_surf_ds, *cloud_surf_ds_w, T_wl.cast<float>());
    pcl::transformPointCloud(*cloud, *cloud_w, T_wl.cast<float>());
    *cloud_surf_map += *cloud_surf_ds_w;
    *cloud_surf_map_global += *cloud_surf_ds_w;

    // Voxel downsampling and reference cloud update
    voxelDownSample(cloud_surf_map, voxel_size);
    voxelDownSample(cloud_surf_map_global, voxel_size);
    cloud_surf_ref = cloud_surf_ds;

    pcl::CropBox<PointType> cropBoxFilter;

    pcl::PointCloud<PointType>::Ptr laserCloudSurfMapLocal(new pcl::PointCloud<PointType>);

    // crop box filter
    float max_distance = 20.0;
    cropBoxFilter.setMin(Eigen::Vector4f((float)T_wl(0, 3) - max_distance, (float)T_wl(1, 3) - max_distance, (float)T_wl(2, 3) - max_distance, 1.0));
    cropBoxFilter.setMax(Eigen::Vector4f((float)T_wl(0, 3) + max_distance, (float)T_wl(1, 3) + max_distance, (float)T_wl(2, 3) + max_distance, 1.0));
    cropBoxFilter.setNegative(false);

    cropBoxFilter.setInputCloud(cloud_surf_map);
    cropBoxFilter.filter(*laserCloudSurfMapLocal);

    cloud_surf_map = laserCloudSurfMapLocal;
}

Eigen::Matrix4d iterativeClosestPoint(const pcl::PointCloud<PointType>::Ptr &surfCloud,
                                      const pcl::PointCloud<PointType>::Ptr &surfMapCloud)
{

    // Build kd-tree of map cloud
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfMap(new pcl::KdTreeFLANN<PointType>());
    kdtreeSurfMap->setInputCloud(surfMapCloud);
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeCurrentCloud(new pcl::KdTreeFLANN<PointType>());

    // normal estimation (current lidar scan)
    pcl::PointCloud<pcl::Normal>::Ptr surfCurrentNormal(new pcl::PointCloud<pcl::Normal>);
    std::vector<bool> planeValid(surfCloud->points.size());
    estimateNormal(surfCloud, surfCurrentNormal, planeValid);

    Eigen::Vector3d deltaTrans(0, 0, 0);
    Eigen::Vector3d deltaAxisAngle(0, 0, 0);

    Eigen::Matrix<double, 6, 1> deltaParam;
    deltaParam.setZero();

    Eigen::Matrix<double, 6, 1> deltaParamRef;
    deltaParamRef.setZero();

    int optimization_count = 100;
    int num_of_cores = 10;

    for (int iterCount = 0; iterCount < optimization_count; iterCount++)
    {

        deltaTrans = deltaParam.block<3, 1>(0, 0);
        deltaAxisAngle = deltaParam.block<3, 1>(3, 0);
        ceres::Problem problem;
        ceres::Solver::Options options;
        ceres::Solver::Summary summary;

        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.minimizer_progress_to_stdout = false;
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        options.num_threads = num_of_cores;
        options.max_num_iterations = 4;

        problem.AddParameterBlock(deltaParam.data(), 6);

        Eigen::Matrix3d delta_R = axisAngleToR(deltaAxisAngle);
        Eigen::Vector3d delta_t = deltaTrans;

        int surf_num = 0;

        std::vector<Eigen::Matrix<double, 1, 6>> Jacobians;
        std::vector<double> bs;

        std::vector<Eigen::Vector3d> ps, qs, nps, nqs;
        std::vector<double> residuals;

        // #pragma omp parallel for num_threads(num_of_cores)
        for (int i = 0; i < (int)surfCloud->points.size(); i++)
        {

            if (!planeValid[i])
                continue;

            Eigen::Vector3d p_orig(surfCloud->points[i].x,
                                   surfCloud->points[i].y,
                                   surfCloud->points[i].z);

            Eigen::Vector3d p_curr = (delta_R * p_orig + delta_t);

            Eigen::Vector3d np(surfCurrentNormal->points[i].normal_x,
                               surfCurrentNormal->points[i].normal_y,
                               surfCurrentNormal->points[i].normal_z);

            pcl::PointXYZI point_temp;
            point_temp.x = p_curr[0];
            point_temp.y = p_curr[1];
            point_temp.z = p_curr[2];

            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            // 현재 위치에서 가장 가까운 map point 다섯개 선택
            // 실제 map point 는 rotation 되어있으나, normal을 일정하게 유지하기 위하여 rotation 되지 않은 원본 point 로 plane normal 을 계산한다.
            // p, np, q, np 구하고 ceres 에 push 해보자!

            kdtreeSurfMap->nearestKSearch(point_temp, 5, pointSearchInd, pointSearchSqDis);

            Eigen::Matrix<double, 5, 3> matA0;
            Eigen::Matrix<double, 5, 1> matB0 = -1 * Eigen::Matrix<double, 5, 1>::Ones();

            if (pointSearchSqDis[4] < 1.0)
            {

                for (int j = 0; j < 5; j++)
                {
                    matA0(j, 0) = surfMapCloud->points[pointSearchInd[j]].x;
                    matA0(j, 1) = surfMapCloud->points[pointSearchInd[j]].y;
                    matA0(j, 2) = surfMapCloud->points[pointSearchInd[j]].z;
                }
                // find the norm of plane
                Eigen::Vector3d nq = matA0.colPivHouseholderQr().solve(matB0);

                nq.normalize();

                Eigen::Vector3d q_orig(surfMapCloud->points[pointSearchInd[0]].x,
                                       surfMapCloud->points[pointSearchInd[0]].y,
                                       surfMapCloud->points[pointSearchInd[0]].z);

                double negative_OA_dot_norm = -(q_orig[0] * nq[0] + q_orig[1] * nq[1] + q_orig[2] * nq[2]);

                bool planeValid = true;

                for (int j = 0; j < 5; j++)
                {
                    // if OX * n > 0.2, then plane is not fit well
                    if (fabs(nq(0) * surfMapCloud->points[pointSearchInd[j]].x +
                             nq(1) * surfMapCloud->points[pointSearchInd[j]].y +
                             nq(2) * surfMapCloud->points[pointSearchInd[j]].z +
                             negative_OA_dot_norm) > 0.2)
                    {
                        planeValid = false;
                        break;
                    }
                }

                if (planeValid)
                {
                    surf_num += 1;
                    ps.emplace_back(p_orig);
                    qs.emplace_back(q_orig);
                    nps.emplace_back(np);
                    nqs.emplace_back(nq);
                    double residual = abs((np + nq).dot(delta_R * p_orig + delta_t - q_orig));
                    residuals.emplace_back(residual);
                }
            }
        }

        if (surf_num < 20)
            std::cout << "Not enough surface scan points!!!" << std::endl;

        else
        {
            // calculate statistics for tuchy loss
            std::sort(residuals.begin(), residuals.end());

            int sizeHalf = (int)(((int)residuals.size()) / 2);
            double medianRes = residuals[sizeHalf];
            double sigma = 4.6851 * 1.4826 * (1.0 + 5.0 / ((int)residuals.size() * 2 - 6)) * sqrt(medianRes);

            ceres::LossFunction *loss_function = new ceres::TukeyLoss(sigma);

            for (int i = 0; i < ps.size(); i++)
            {
                ceres::CostFunction *cost_function = new SurfSymmAnalyticCostFunction(ps[i], nps[i], qs[i], nqs[i]);
                problem.AddResidualBlock(cost_function, loss_function, deltaParam.data());
            }

            ceres::Solve(options, &problem, &summary);
            // std::cout << summary.BriefReport() << std::endl;
        }

        if ((deltaParamRef - deltaParam).block<3, 1>(0, 0).norm() < 0.01 &&
            (deltaParamRef - deltaParam).block<3, 1>(3, 0).norm() < 0.01)
        {
            deltaParam = deltaParamRef;
            break;
        }

        deltaParamRef = deltaParam;
    }

    // After optimization,
    Eigen::Matrix4d T_rel = Eigen::Matrix4d::Identity();
    T_rel.block<3, 3>(0, 0) = axisAngleToR(deltaParam.block<3, 1>(3, 0));
    T_rel.block<3, 1>(0, 3) = deltaParam.block<3, 1>(0, 0);

    return T_rel;
}

void LidarHandler::featureExtraction(const pcl::PointCloud<PointType>::Ptr &pc_in, pcl::PointCloud<PointType>::Ptr &pc_out_surf)
{

    int N_SCANS = 128;
    std::vector<pcl::PointCloud<PointType>::Ptr> laserCloudScans;

    for (int i = 0; i < N_SCANS; i++)
    {
        laserCloudScans.push_back(pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>()));
    }

    // #pragma omp parallel for num_threads(4)
    for (int i = 0; i < (int)pc_in->points.size(); i++)
    {
        int scanID = 0;
        double distance = sqrt(pc_in->points[i].x * pc_in->points[i].x + pc_in->points[i].y * pc_in->points[i].y);
        if (distance > 60.0 || distance < 0.3)
            continue;
        double angle = atan(pc_in->points[i].z / distance) * 180 / M_PI;

        if (N_SCANS == 16)
        {
            scanID = int((angle + 15) / 2 + 0.5);
            if (scanID > (N_SCANS - 1) || scanID < 0)
            {
                continue;
            }
        }
        else if (N_SCANS == 32)
        {
            // Original
            // scanID = int((angle + 92.0 / 3.0) * 3.0 / 4.0);
            // if (scanID > (N_SCANS - 1) || scanID < 0)
            // {
            //     continue;
            // }

            // VLP 32-C
            scanID = int(((angle + 25.0) / 40.0 * 32.0) - 0.5);
            if (scanID > (N_SCANS - 1) || scanID < 0)
            {
                continue;
            }
        }
        else if (N_SCANS == 64)
        {
            // // For New Colledge Stereo // 64-channel / 33.2 FOV
            scanID = int((angle + 16.6) / 0.51875);

            if (scanID > (N_SCANS - 1) || scanID < 0)
            {
                continue;
            }

            // // For KITTI
            // if (angle >= -8.83)
            //     scanID = int((2 - angle) * 3.0 + 0.5);
            // else
            //     scanID = N_SCANS / 2 + int((-8.83 - angle) * 2.0 + 0.5);

            // if (angle > 2 || angle < -24.33 || scanID > 63 || scanID < 0)
            // {
            //     continue;
            // }
        }

        else if (N_SCANS == 128)
        {
            scanID = int((angle + 45) / 2 + 0.5);

            if (scanID > (N_SCANS - 1) || scanID < 0)
            {
                continue;
            }
        }
        else
        {
            printf("wrong scan number\n");
        }

        laserCloudScans[scanID]->push_back(pc_in->points[i]);
    }

    for (int i = 0; i < N_SCANS; i++)
    {

        if (laserCloudScans[i]->points.size() < 131)
        {
            // std::cout<<"Not enough line pts size // "<<laserCloudScans[i]->points.size()<<std::endl;
            continue;
        }
        else
        {
            // std::cout<<"Enough line pts size // "<<laserCloudScans[i]->points.size()<<std::endl;
        }

        std::vector<Double2d> cloudCurvature;
        int total_points = laserCloudScans[i]->points.size() - 10;

        for (int j = 5; j < (int)laserCloudScans[i]->points.size() - 5; j++)
        {
            double diffX = laserCloudScans[i]->points[j - 5].x + laserCloudScans[i]->points[j - 4].x + laserCloudScans[i]->points[j - 3].x + laserCloudScans[i]->points[j - 2].x + laserCloudScans[i]->points[j - 1].x - 10 * laserCloudScans[i]->points[j].x + laserCloudScans[i]->points[j + 1].x + laserCloudScans[i]->points[j + 2].x + laserCloudScans[i]->points[j + 3].x + laserCloudScans[i]->points[j + 4].x + laserCloudScans[i]->points[j + 5].x;
            double diffY = laserCloudScans[i]->points[j - 5].y + laserCloudScans[i]->points[j - 4].y + laserCloudScans[i]->points[j - 3].y + laserCloudScans[i]->points[j - 2].y + laserCloudScans[i]->points[j - 1].y - 10 * laserCloudScans[i]->points[j].y + laserCloudScans[i]->points[j + 1].y + laserCloudScans[i]->points[j + 2].y + laserCloudScans[i]->points[j + 3].y + laserCloudScans[i]->points[j + 4].y + laserCloudScans[i]->points[j + 5].y;
            double diffZ = laserCloudScans[i]->points[j - 5].z + laserCloudScans[i]->points[j - 4].z + laserCloudScans[i]->points[j - 3].z + laserCloudScans[i]->points[j - 2].z + laserCloudScans[i]->points[j - 1].z - 10 * laserCloudScans[i]->points[j].z + laserCloudScans[i]->points[j + 1].z + laserCloudScans[i]->points[j + 2].z + laserCloudScans[i]->points[j + 3].z + laserCloudScans[i]->points[j + 4].z + laserCloudScans[i]->points[j + 5].z;
            Double2d distance(j, diffX * diffX + diffY * diffY + diffZ * diffZ);
            cloudCurvature.push_back(distance);
        }

        // #pragma omp parallel for num_threads(4)
        for (int j = 0; j < 6; j++)
        {
            int sector_length = (int)(total_points / 6);
            int sector_start = sector_length * j;
            int sector_end = sector_length * (j + 1) - 1;
            if (j == 5)
            {
                sector_end = total_points - 1;
            }

            std::vector<Double2d> subCloudCurvature(cloudCurvature.begin() + sector_start, cloudCurvature.begin() + sector_end);
            featureExtractionFromSector(laserCloudScans[i], subCloudCurvature, pc_out_surf);
        }
    }
}

void LidarHandler::featureExtractionFromSector(const pcl::PointCloud<PointType>::Ptr &pc_in, std::vector<Double2d> &cloudCurvature, pcl::PointCloud<PointType>::Ptr &pc_out_surf)
{

    std::sort(cloudCurvature.begin(), cloudCurvature.end(), [](const Double2d &a, const Double2d &b)
              { return a.value < b.value; });

    int largestPickedNum = 0;
    std::vector<int> picked_points;
    int point_info_count = 0;

    ////#pragma omp parallel for num_threads(4)
    for (int i = cloudCurvature.size() - 1; i >= 0; i--)
    {
        int ind = cloudCurvature[i].id;
        if (std::find(picked_points.begin(), picked_points.end(), ind) == picked_points.end())
        {
            if (cloudCurvature[i].value <= 0.1)
            {
                break;
            }

            largestPickedNum++;
            picked_points.push_back(ind);

            if (largestPickedNum <= 20)
            {
                point_info_count++;
            }
            else
            {
                break;
            }

            for (int k = 1; k <= 5; k++)
            {
                double diffX = pc_in->points[ind + k].x - pc_in->points[ind + k - 1].x;
                double diffY = pc_in->points[ind + k].y - pc_in->points[ind + k - 1].y;
                double diffZ = pc_in->points[ind + k].z - pc_in->points[ind + k - 1].z;
                if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                {
                    break;
                }
                picked_points.push_back(ind + k);
            }
            for (int k = -1; k >= -5; k--)
            {
                double diffX = pc_in->points[ind + k].x - pc_in->points[ind + k + 1].x;
                double diffY = pc_in->points[ind + k].y - pc_in->points[ind + k + 1].y;
                double diffZ = pc_in->points[ind + k].z - pc_in->points[ind + k + 1].z;
                if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                {
                    break;
                }
                picked_points.push_back(ind + k);
            }
        }
    }

    // #pragma omp parallel for num_threads(4)
    for (int i = 0; i <= (int)cloudCurvature.size() - 1; i++)
    {
        int ind = cloudCurvature[i].id;
        if (std::find(picked_points.begin(), picked_points.end(), ind) == picked_points.end() && cloudCurvature[i].value < 0.2)
        {
            Eigen::Vector3d pointXYZ(pc_in->points[ind].x, pc_in->points[ind].y, pc_in->points[ind].z);
            double distance = pointXYZ.norm();

            if (distance > 0.5)
                pc_out_surf->push_back(pc_in->points[ind]);
        }
    }
}

Double2d::Double2d(int id_in, double value_in)
{
    id = id_in;
    value = value_in;
};

PointsInfo::PointsInfo(int layer_in, double time_in)
{
    layer = layer_in;
    time = time_in;
};
