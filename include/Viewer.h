//
// Created by gaoxiang on 19-5-4.
//

#ifndef MYSLAM_VIEWER_H
#define MYSLAM_VIEWER_H

#include <thread>
#include <pangolin/pangolin.h>

#include <Common.h>
#include <GeometryUtils.h>
#include <DataStructure.h>

class Viewer
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Viewer> Ptr;

    Viewer();

    // void UpdateMap();

    void spinOnce();
    void initialize();

    void drawPoints(std::vector<Eigen::Vector3d> &points, std::vector<Eigen::Vector3d> &colors, int size);
    void drawPoints(pcl::PointCloud<PointType>::Ptr &cloud, Eigen::Vector3d &colors, int size);
    void drawTrajectory(Eigen::Vector3d color, std::vector<Eigen::Matrix4d> trajectory, int size);
    void drawAxis(Eigen::Matrix4d pose);
    void drawAxis(Eigen::Matrix4d pose, Eigen::Vector3d &colors);
    void drawKeyFrame(Frame::Ptr frame);
    void drawFrame(Frame::Ptr frame);

    std::vector<Frame::Ptr> keyframes;
    Frame::Ptr curr_frame;

    std::mutex viewer_data_mutex_;
    pangolin::View vis_display;
    pangolin::OpenGlRenderState vis_camera;

    int id;
    bool flagDrawCurrentPointCloud = true;
    bool flagDrawMesh = false;
    bool start = false;

    int width, height;
    double fx, fy, cx, cy;

    std::vector<Eigen::Vector3d> current_points, current_colors;
    std::vector<Eigen::Vector3d> ref_points, ref_colors;
    std::vector<Eigen::Vector3d> ref_points_before, ref_colors_before;
    std::vector<Eigen::Vector3d> map_points, map_colors;
    std::vector<Eigen::Matrix4d> trajectory, trajectory_imu;

    pcl::PointCloud<PointType>::Ptr cloud_surf;
    pcl::PointCloud<PointType>::Ptr cloud_map;
    pcl::PointCloud<PointType>::Ptr cloud_map_local;
    pcl::PointCloud<PointType>::Ptr cloud_cumulated;
    pcl::PointCloud<PointType>::Ptr cloud_feature;

    Eigen::Matrix4d T_wl = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d T_wc = Eigen::Matrix4d::Identity();

    // Topology
    std::vector<Vertex::Ptr> local_vertices;
    std::vector<Edge::Ptr> local_edges;
    std::vector<Triangle::Ptr> local_triangles;

    // line intersection debug
    std::vector<Eigen::Vector3d> line_debug;

    void drawTriangles(const std::vector<Triangle::Ptr> &triangles);

    void setLocalGeometry(const std::vector<Vertex::Ptr> &vertices,
                          const std::vector<Edge::Ptr> &edges,
                          const std::vector<Triangle::Ptr> &triangles)
    {
        local_vertices = vertices;
        local_edges = edges;
        local_triangles = triangles;
    };

    void drawAxisThin(Eigen::Matrix4d pose, Eigen::Vector3d &colors);

};

#endif
