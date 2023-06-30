#pragma once
#ifndef LIDAR_HANDLER_H
#define LIDAR_HANDLER_H

#include <pcl/common/transforms.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/icp.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/crop_box.h>
#include <pcl/registration/correspondence_rejection_surface_normal.h>
#include <pcl/registration/correspondence_estimation_normal_shooting.h>
#include <pcl/features/normal_3d.h>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <GeometryUtils.h>
#include <Config.h>

// Obejctive Functions

Eigen::Matrix<double, 3, 3> skew(Eigen::Matrix<double, 3, 1> mat_in);
Eigen::Matrix<double, 3, 3> axisAngleToR(const Eigen::Vector3d &axisAngle);

class SurfSymmAnalyticCostFunction : public ceres::SizedCostFunction<1, 6>
{
public:
    SurfSymmAnalyticCostFunction(Eigen::Vector3d p_, Eigen::Vector3d np_, Eigen::Vector3d q_, Eigen::Vector3d nq_);
    virtual ~SurfSymmAnalyticCostFunction() {}
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    Eigen::Vector3d p, np, q, nq;
};

class PoseSE3SymmParameterization : public ceres::LocalParameterization
{
public:
    PoseSE3SymmParameterization() {}
    virtual ~PoseSE3SymmParameterization() {}
    virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const;
    virtual bool ComputeJacobian(const double *x, double *jacobian) const;
    virtual int GlobalSize() const { return 6; }
    virtual int LocalSize() const { return 6; }
};

/////////////////////////////////////////////////////////////////

// points covariance class
class Double2d
{
public:
    int id;
    double value;
    Double2d(int id_in, double value_in);
};
// points info class
class PointsInfo
{
public:
    int layer;
    double time;
    PointsInfo(int layer_in, double time_in);
};

/////////////////////////////////////////////////////////////////

class LidarHandler
{
public:
    Eigen::Matrix3d K;
    Eigen::Matrix4d T_il;
    Eigen::Matrix4d T_ic;
    Eigen::Matrix4d T_cl;
    Eigen::Matrix4d T_wl;

    double maxDistanceToPlane;
    double max_distance;
    double min_distance;
    double distance_threshold;
    double distance_ratio_threshold;
    double angle_thresh_max; // / 36 * 35;
    double angle_thresh_min; // M_PI / 36;
    int image_width;
    int image_height;

    bool lidarInit = false;

    double voxel_size;
    // cloudXYZIT::Ptr cloud_surf_ref(new cloudXYZIT);
    // cloudXYZ::Ptr cloud_surf_map(new cloudXYZ);

    pcl::PointCloud<PointType>::Ptr cloud_surf_ref;
    pcl::PointCloud<PointType>::Ptr cloud_surf_w;
    pcl::PointCloud<PointType>::Ptr cloud_surf_map;
    pcl::PointCloud<PointType>::Ptr cloud_surf_map_global;
    pcl::PointCloud<PointType>::Ptr cloud_surf_map_frustum;

    std::deque<std::vector<Eigen::Vector3d>> cumulated_cloud_queue;
    std::vector<Triangle::Ptr> local_triangles_current;
    pcl::CropBox<pcl::PointXYZ> cropBoxFilter;

    Eigen::Matrix4d T_odom_ref;

    //////////////////////////////////////////////////////////////////////////

    LidarHandler(std::string config_path)
    {
        Config::SetParameterFile(config_path);
        maxDistanceToPlane = Config::Get<double>("max_distance_plane");
        max_distance = Config::Get<double>("max_distance");
        min_distance = Config::Get<double>("min_distance");
        distance_threshold = Config::Get<double>("distance_threshold");
        distance_ratio_threshold = Config::Get<double>("distance_ratio_threshold");
        angle_thresh_max = M_PI; // / 36 * 35;
        angle_thresh_min = 0;    // M_PI / 36;
        image_width = Config::Get<int>("image_width");
        image_height = Config::Get<int>("image_height");

        K << 872.614, 0, 615.868, 0, 872.992, 552.648, 0.0, 0.0, 1.0;
        T_il << -0.99999353, 0.00169655, 0.00327721, -0.0554467, 0.00172166, 0.99996872, 0.00767292, 0.00517844, -0.00326409, 0.0076785, -0.99996512, 0.03128449, 0, 0, 0, 1;
        T_ic << -0.00700046, 0.00632448, -0.9999555, -0.02536939, -0.99989, -0.01312013, 0.00691702, 0.02244737, -0.0130758, 0.99989393, 0.00641563, -0.02027493, 0, 0, 0, 1;
        T_cl = T_ic.inverse() * T_il;

        T_odom_ref = Eigen::Matrix4d::Identity();

        T_wl = Eigen::Matrix4d::Identity();

        voxel_size = Config::Get<double>("voxel_size");

        cloud_surf_ref = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);
        cloud_surf_map = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);
        cloud_surf_map_global = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);
        cloud_surf_w = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);
        cloud_surf_map_frustum = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);
        // cloud_w = cloudXYZ::Ptr(new cloudXYZ);
    }

    void featureExtraction(const pcl::PointCloud<PointType>::Ptr &pc_in, pcl::PointCloud<PointType>::Ptr &pc_out_surf);
    void featureExtractionFromSector(const pcl::PointCloud<PointType>::Ptr &pc_in, std::vector<Double2d> &cloudCurvature, pcl::PointCloud<PointType>::Ptr &pc_out_surf);
    void updateOdometry(const pcl::PointCloud<PointType>::Ptr &cloud, const bool fix);
};

void convertPointType(const cloudXYZIT::Ptr &cloud_in, const cloudXYZ::Ptr &cloud_out);
void voxelDownSample(pcl::PointCloud<PointType>::Ptr &cloud_in, const double voxel_size);
void voxelDownSample(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud_in, const double voxel_size);


Eigen::Matrix4d iterativeClosestPoint(const pcl::PointCloud<PointType>::Ptr &surfCloud,
                                      const pcl::PointCloud<PointType>::Ptr &surfMapCloud);

void estimateNormal(const pcl::PointCloud<PointType>::Ptr &src,
                    const pcl::PointCloud<pcl::Normal>::Ptr &cloud_normal,
                    std::vector<bool> &planeValid);

#endif