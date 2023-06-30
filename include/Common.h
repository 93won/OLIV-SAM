#pragma once
#ifndef COMMON_H
#define COMMON_H

// std
#include <atomic>
#include <condition_variable>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <typeinfo>
#include <unordered_map>
#include <vector>
#include <fstream> // ifstream header
#include <sstream>
#include <chrono>
#include <limits>
#include <random>
#include <numeric>
#include <algorithm>
#include <queue>
#include <deque>
#include <unordered_map>
#include <unordered_set>


// eigen
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>

// opencv
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
// glog
#include <glog/logging.h>

// pcl
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>




// typedefs for eigen
typedef Eigen::Matrix<double, 3, 3> Mat33;
typedef Eigen::Matrix<double, 3, 4> Mat34;
typedef Eigen::Matrix<double, 4, 4> Mat44;

typedef Eigen::Vector2d Vec2;
typedef Eigen::Vector3d Vec3;
typedef Eigen::Vector4d Vec4;
typedef Eigen::Matrix<double, 5, 1> Vec5;
typedef Eigen::Matrix<double, 6, 1> Vec6;
typedef Eigen::Matrix<double, 7, 1> Vec7;


typedef pcl::PointXYZI PointType;


enum class TrackerStatus
{
    INITING,
    Tracker_GOOD,
    Tracker_BAD,
    LOST
};

struct EIGEN_ALIGN16 PointXYZIT
{
    PCL_ADD_POINT4D;
    PCL_ADD_INTENSITY;
    double offset_time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIT,
                                  (double, x, x)(double, y, y)(double, z, z)(double, intensity, intensity)(double, offset_time, offset_time))

typedef pcl::PointCloud<pcl::PointXYZ> cloudXYZ;
typedef pcl::PointCloud<PointXYZIT> cloudXYZIT;

// split function

static std::vector<std::string> split(std::string input, char delimiter)
{
    std::vector<std::string> answer;
    std::stringstream ss(input);
    std::string temp;

    while (getline(ss, temp, delimiter))
    {
        answer.push_back(temp);
    }

    return answer;
}

static std::vector<size_t> argsort_f(const std::vector<double> &v)
{

    std::vector<size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);

    std::stable_sort(idx.begin(), idx.end(),
                     [&v](size_t i1, size_t i2)
                     { return v[i1] < v[i2]; });

    return idx;
}

static std::vector<size_t> argsort_d(const std::vector<double> &v)
{

    std::vector<size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);

    std::stable_sort(idx.begin(), idx.end(),
                     [&v](size_t i1, size_t i2)
                     { return v[i1] < v[i2]; });

    return idx;
}

static std::vector<size_t> argsort_i(const std::vector<int> &v)
{

    std::vector<size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);

    std::stable_sort(idx.begin(), idx.end(),
                     [&v](size_t i1, size_t i2)
                     { return v[i1] < v[i2]; });

    return idx;
}


class Point2D : public std::array<float, 2>
{
public:
    // dimension of space (or "k" of k-d tree)
    // KDTree class accesses this member
    static const int DIM = 2;

    // the constructors
    Point2D() {}
    Point2D(float x, float y)
    {
        (*this)[0] = x;
        (*this)[1] = y;
    }

    // conversion to OpenCV Point2d
    operator cv::Point2d() const { return cv::Point2d((*this)[0], (*this)[1]); }
};

//////////////////////////////////////////

// Messages
struct ImuMessage
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    Eigen::Vector3d linear_acc;
    Eigen::Vector3d angular_vel;

    double timestamp;

    ImuMessage(const Eigen::Vector3d linear_acc_, const Eigen::Vector3d angular_vel_, const double timestamp_)
    {
        linear_acc = linear_acc_;
        angular_vel = angular_vel_;
        timestamp = timestamp_;
    }
};

// Messages
struct LidarMessage
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    std::string lidar_path;
    double timestamp;

    LidarMessage(const std::string lidar_path_, const double timestamp_)
    {
        lidar_path = lidar_path_;
        timestamp = timestamp_;
    }
};

struct ImgMessage
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    std::string img_path;
    double timestamp;

    ImgMessage(const std::string img_path_,  const double timestamp_)
    {
        img_path = img_path_;
        timestamp = timestamp_;
    }
};

static Eigen::Matrix<double, 3, 3> skew_symm(Eigen::Matrix<double, 3, 1> mat_in)
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


static void deskewCloud(const cloudXYZIT::Ptr &cloud, const Eigen::Matrix4d T_ref, const Eigen::Matrix4d T_curr)
{
    Eigen::Matrix4d T_rel = T_ref.inverse() * T_curr;
    Eigen::Quaterniond q_wl_rel_start(1, 0, 0, 0);
    Eigen::Quaterniond q_wl_rel_end(T_rel.block<3, 3>(0, 0));

    // de-skew point
    for (auto &p : cloud->points)
    {
        double offset_time = p.offset_time;

        // local->global
        Eigen::Vector3d xyz_local(p.x, p.y, p.z);

        Eigen::Quaterniond q_interpolated = q_wl_rel_start.slerp(offset_time, q_wl_rel_end);
        Eigen::Matrix3d R_interpolated = q_interpolated.normalized().toRotationMatrix();
        Eigen::Vector3d t_interpolated = T_rel.block<3, 1>(0, 3) * offset_time;

        Eigen::Matrix4d T_interpolated = Eigen::Matrix4d::Identity();
        T_interpolated.block<3, 3>(0, 0) = R_interpolated;
        T_interpolated.block<3, 1>(0, 3) = t_interpolated;

        Eigen::Matrix4d T_interpolated_wl = T_ref * T_interpolated;
        Eigen::Vector3d xyz_world = T_interpolated_wl.block<3, 3>(0, 0) * xyz_local + T_interpolated_wl.block<3, 1>(0, 3);

        // global->local
        xyz_local = T_ref.inverse().block<3, 3>(0, 0) * xyz_world + T_ref.inverse().block<3, 1>(0, 3);

        p.x = xyz_local[0];
        p.y = xyz_local[1];
        p.z = xyz_local[2];
    }
}


#ifndef M_PI
// M_PI is not part of the C++ standard. Rather it is part of the POSIX standard. As such,
// it is not directly available on Visual C++ (although _USE_MATH_DEFINES does exist).
#define M_PI 3.14159265358979323846
#endif

typedef float real_t;

// Eigen matrix types

template<size_t R, size_t C>
using MatRC_t = Eigen::Matrix<double, R, C>;

using Mat22_t = Eigen::Matrix2d;

using Mat33_t = Eigen::Matrix3d;

using Mat44_t = Eigen::Matrix4d;

using Mat55_t = MatRC_t<5, 5>;

using Mat66_t = MatRC_t<6, 6>;

using Mat77_t = MatRC_t<7, 7>;

using Mat34_t = MatRC_t<3, 4>;

using MatX_t = Eigen::MatrixXd;

// Eigen vector types

template<size_t R>
using VecR_t = Eigen::Matrix<double, R, 1>;

using Vec2_t = Eigen::Vector2d;

using Vec3_t = Eigen::Vector3d;

using Vec4_t = Eigen::Vector4d;

using Vec5_t = VecR_t<5>;

using Vec6_t = VecR_t<6>;

using Vec7_t = VecR_t<7>;

using VecX_t = Eigen::VectorXd;

// Eigen Quaternion type

using Quat_t = Eigen::Quaterniond;

// STL with Eigen custom allocator

template<typename T>
using eigen_alloc_vector = std::vector<T, Eigen::aligned_allocator<T>>;

template<typename T, typename U>
using eigen_alloc_map = std::map<T, U, std::less<T>, Eigen::aligned_allocator<std::pair<const T, U>>>;

template<typename T>
using eigen_alloc_set = std::set<T, std::less<T>, Eigen::aligned_allocator<const T>>;

template<typename T, typename U>
using eigen_alloc_unord_map = std::unordered_map<T, U, std::hash<T>, std::equal_to<T>, Eigen::aligned_allocator<std::pair<const T, U>>>;

template<typename T>
using eigen_alloc_unord_set = std::unordered_set<T, std::hash<T>, std::equal_to<T>, Eigen::aligned_allocator<const T>>;

// vector operators

template<typename T>
inline Vec2_t operator+(const Vec2_t& v1, const cv::Point_<T>& v2) {
    return {v1(0) + v2.x, v1(1) + v2.y};
}

template<typename T>
inline Vec2_t operator+(const cv::Point_<T>& v1, const Vec2_t& v2) {
    return v2 + v1;
}

template<typename T>
inline Vec2_t operator-(const Vec2_t& v1, const cv::Point_<T>& v2) {
    return v1 + (-v2);
}

template<typename T>
inline Vec2_t operator-(const cv::Point_<T>& v1, const Vec2_t& v2) {
    return v1 + (-v2);
}



#endif