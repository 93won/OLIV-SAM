#pragma once
#ifndef GEOMETRY_UTILS_H
#define GEOMETRY_UTILS_H

#include <Common.h>
#include <unordered_map>

#include <vector>
#include <opencv2/opencv.hpp>

#include <vector>
#include <numeric>
#include <algorithm>
#include <exception>
#include <functional>

namespace kdt
{
    /** @brief k-d tree class.
     */
    template <class PointT>
    class KDTree
    {
    public:
        /** @brief The constructors.
         */
        KDTree() : root_(nullptr){};
        KDTree(const std::vector<PointT> &points) : root_(nullptr) { build(points); }

        /** @brief The destructor.
         */
        ~KDTree() { clear(); }

        /** @brief Re-builds k-d tree.
         */
        void build(const std::vector<PointT> &points)
        {
            clear();

            points_ = points;

            std::vector<int> indices(points.size());
            std::iota(std::begin(indices), std::end(indices), 0);

            root_ = buildRecursive(indices.data(), (int)points.size(), 0);
        }

        /** @brief Clears k-d tree.
         */
        void clear()
        {
            clearRecursive(root_);
            root_ = nullptr;
            points_.clear();
        }

        /** @brief Validates k-d tree.
         */
        bool validate() const
        {
            try
            {
                validateRecursive(root_, 0);
            }
            catch (const Exception &)
            {
                return false;
            }

            return true;
        }

        /** @brief Searches the nearest neighbor.
         */
        int nnSearch(const PointT &query, double *minDist = nullptr) const
        {
            int guess;
            double _minDist = std::numeric_limits<double>::max();

            nnSearchRecursive(query, root_, &guess, &_minDist);

            if (minDist)
                *minDist = _minDist;

            return guess;
        }

        /** @brief Searches k-nearest neighbors.
         */
        std::vector<int> knnSearch(const PointT &query, int k) const
        {
            KnnQueue queue(k);
            knnSearchRecursive(query, root_, queue, k);

            std::vector<int> indices(queue.size());
            for (size_t i = 0; i < queue.size(); i++)
                indices[i] = queue[i].second;

            return indices;
        }

        /** @brief Searches neighbors within radius.
         */
        std::vector<int> radiusSearch(const PointT &query, double radius) const
        {
            std::vector<int> indices;
            radiusSearchRecursive(query, root_, indices, radius);
            return indices;
        }

    private:
        /** @brief k-d tree node.
         */
        struct Node
        {
            int idx;       //!< index to the original point
            Node *next[2]; //!< pointers to the child nodes
            int axis;      //!< dimension's axis

            Node() : idx(-1), axis(-1) { next[0] = next[1] = nullptr; }
        };

        /** @brief k-d tree exception.
         */
        class Exception : public std::exception
        {
            using std::exception::exception;
        };

        /** @brief Bounded priority queue.
         */
        template <class T, class Compare = std::less<T>>
        class BoundedPriorityQueue
        {
        public:
            BoundedPriorityQueue() = delete;
            BoundedPriorityQueue(size_t bound) : bound_(bound) { elements_.reserve(bound + 1); };

            void push(const T &val)
            {
                auto it = std::find_if(std::begin(elements_), std::end(elements_),
                                       [&](const T &element)
                                       { return Compare()(val, element); });
                elements_.insert(it, val);

                if (elements_.size() > bound_)
                    elements_.resize(bound_);
            }

            const T &back() const { return elements_.back(); };
            const T &operator[](size_t index) const { return elements_[index]; }
            size_t size() const { return elements_.size(); }

        private:
            size_t bound_;
            std::vector<T> elements_;
        };

        /** @brief Priority queue of <distance, index> pair.
         */
        using KnnQueue = BoundedPriorityQueue<std::pair<double, int>>;

        /** @brief Builds k-d tree recursively.
         */
        Node *buildRecursive(int *indices, int npoints, int depth)
        {
            if (npoints <= 0)
                return nullptr;

            const int axis = depth % PointT::DIM;
            const int mid = (npoints - 1) / 2;

            std::nth_element(indices, indices + mid, indices + npoints, [&](int lhs, int rhs)
                             { return points_[lhs][axis] < points_[rhs][axis]; });

            Node *node = new Node();
            node->idx = indices[mid];
            node->axis = axis;

            node->next[0] = buildRecursive(indices, mid, depth + 1);
            node->next[1] = buildRecursive(indices + mid + 1, npoints - mid - 1, depth + 1);

            return node;
        }

        /** @brief Clears k-d tree recursively.
         */
        void clearRecursive(Node *node)
        {
            if (node == nullptr)
                return;

            if (node->next[0])
                clearRecursive(node->next[0]);

            if (node->next[1])
                clearRecursive(node->next[1]);

            delete node;
        }

        /** @brief Validates k-d tree recursively.
         */
        void validateRecursive(const Node *node, int depth) const
        {
            if (node == nullptr)
                return;

            const int axis = node->axis;
            const Node *node0 = node->next[0];
            const Node *node1 = node->next[1];

            if (node0 && node1)
            {
                if (points_[node->idx][axis] < points_[node0->idx][axis])
                    throw Exception();

                if (points_[node->idx][axis] > points_[node1->idx][axis])
                    throw Exception();
            }

            if (node0)
                validateRecursive(node0, depth + 1);

            if (node1)
                validateRecursive(node1, depth + 1);
        }

        static double distance(const PointT &p, const PointT &q)
        {
            double dist = 0;
            for (size_t i = 0; i < PointT::DIM; i++)
                dist += (p[i] - q[i]) * (p[i] - q[i]);
            return sqrt(dist);
        }

        /** @brief Searches the nearest neighbor recursively.
         */
        void nnSearchRecursive(const PointT &query, const Node *node, int *guess, double *minDist) const
        {
            if (node == nullptr)
                return;

            const PointT &train = points_[node->idx];

            const double dist = distance(query, train);
            if (dist < *minDist)
            {
                *minDist = dist;
                *guess = node->idx;
            }

            const int axis = node->axis;
            const int dir = query[axis] < train[axis] ? 0 : 1;
            nnSearchRecursive(query, node->next[dir], guess, minDist);

            const double diff = fabs(query[axis] - train[axis]);
            if (diff < *minDist)
                nnSearchRecursive(query, node->next[!dir], guess, minDist);
        }

        /** @brief Searches k-nearest neighbors recursively.
         */
        void knnSearchRecursive(const PointT &query, const Node *node, KnnQueue &queue, int k) const
        {
            if (node == nullptr)
                return;

            const PointT &train = points_[node->idx];

            const double dist = distance(query, train);
            queue.push(std::make_pair(dist, node->idx));

            const int axis = node->axis;
            const int dir = query[axis] < train[axis] ? 0 : 1;
            knnSearchRecursive(query, node->next[dir], queue, k);

            const double diff = fabs(query[axis] - train[axis]);
            if ((int)queue.size() < k || diff < queue.back().first)
                knnSearchRecursive(query, node->next[!dir], queue, k);
        }

        /** @brief Searches neighbors within radius.
         */
        void radiusSearchRecursive(const PointT &query, const Node *node, std::vector<int> &indices, double radius) const
        {
            if (node == nullptr)
                return;

            const PointT &train = points_[node->idx];

            const double dist = distance(query, train);
            if (dist < radius)
                indices.push_back(node->idx);

            const int axis = node->axis;
            const int dir = query[axis] < train[axis] ? 0 : 1;
            radiusSearchRecursive(query, node->next[dir], indices, radius);

            const double diff = fabs(query[axis] - train[axis]);
            if (diff < radius)
                radiusSearchRecursive(query, node->next[!dir], indices, radius);
        }

        Node *root_;                 //!< root node
        std::vector<PointT> points_; //!< points
    };
} // kdt

class Subdiv2DIndex : public cv::Subdiv2D
{
public:
    Subdiv2DIndex(cv::Rect rectangle) : Subdiv2D{rectangle} {}

    // Source code of Subdiv2D: https://github.com/opencv/opencv/blob/master/modules/imgproc/src/subdivision2d.cpp#L762
    // The implementation tweaks getTrianglesList() so that only the indice of the triangle inside the image are returned
    void getTrianglesIndices(std::vector<std::vector<int>> &triangleList)
    {
        triangleList.clear();
        int i, total = (int)(qedges.size() * 4);
        std::vector<bool> edgemask(total, false);
        const bool filterPoints = true;
        cv::Rect2f rect(topLeft.x, topLeft.y, bottomRight.x - topLeft.x, bottomRight.y - topLeft.y);

        for (i = 4; i < total; i += 2)
        {
            if (edgemask[i])
                continue;
            cv::Point2f a, b, c;
            int edge_a = i;
            int indexA = edgeOrg(edge_a, &a) - 4;
            if (filterPoints && !rect.contains(a))
                continue;
            int edge_b = getEdge(edge_a, NEXT_AROUND_LEFT);
            int indexB = edgeOrg(edge_b, &b) - 4;
            if (filterPoints && !rect.contains(b))
                continue;
            int edge_c = getEdge(edge_b, NEXT_AROUND_LEFT);
            int indexC = edgeOrg(edge_c, &c) - 4;
            if (filterPoints && !rect.contains(c))
                continue;
            edgemask[edge_a] = true;
            edgemask[edge_b] = true;
            edgemask[edge_c] = true;

            std::vector<int> triangle = {indexA, indexB, indexC};

            triangleList.push_back(triangle);
        }
    }
};

struct Vertex;
struct Edge;
struct Triangle;
struct Plane;
struct RoomPlan;

struct Vertex
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Vertex> Ptr;
    Eigen::Vector3d xyz;
    Eigen::Vector3d xyz_orig;
    int id;
    std::string key;
    bool valid;
    Eigen::Vector3d normal;

    bool isProjected;
    int cluster_id;

    //////////////////////////////////////////////////
    bool isLocked;
    std::shared_ptr<Plane> minMaxPlane;
    std::shared_ptr<Vertex> constraintVertex;
    //////////////////////////////////////////////////

    /////////////////////////////////////////////////
    double offset_time;
    double intensity;
    /////////////////////////////////////////////////

    std::vector<std::shared_ptr<Edge>> edges;
    std::vector<std::shared_ptr<Triangle>> triangles;
    std::vector<std::shared_ptr<Vertex>> vertices_adj;

    std::unordered_map<std::string, std::shared_ptr<Triangle>> hash_map_triangle;

    Vertex(const Eigen::Vector3d &xyz_, const int id_)
    {
        xyz = xyz_;
        xyz_orig = xyz_;
        id = id_;
        key = "v_" + std::to_string(id);
        valid = false;
        isProjected = false;
        cluster_id = -1;
        isLocked = false;
    }

    Vertex(const Eigen::Vector3d &xyz_, const int id_, const double offset_time_, double intensity_)
    {
        xyz = xyz_;
        xyz_orig = xyz_;
        id = id_;
        key = "v_" + std::to_string(id);
        valid = false;
        isProjected = false;
        cluster_id = -1;
        isLocked = false;
        offset_time = offset_time_;
        intensity = intensity_;
    }

    void addEdge(const std::shared_ptr<Edge> &edge)
    {
        edges.push_back(edge);
    }

    void addTriangle(const std::shared_ptr<Triangle> &triangle, const std::string key)
    {
        if (hash_map_triangle.find(key) == hash_map_triangle.end())
        {
            hash_map_triangle.insert(std::make_pair(key, triangle));
            triangles.push_back(triangle);
        }
    }

    void evaluateVertexNormal()
    {
        std::vector<Eigen::Vector3d> v_adj;

        // get connected vertices (in local coordinate)
        for (auto &vertex : vertices_adj)
            v_adj.push_back(vertex->xyz - xyz);

        Eigen::MatrixXd A; // (N,3)
        Eigen::MatrixXd b; // (N,1)

        A.resize((int)v_adj.size(), 3);
        b.resize((int)v_adj.size(), 1);

        for (size_t i = 0; i < v_adj.size(); i++)
        {
            A(i, 0) = v_adj[i][0];
            A(i, 1) = v_adj[i][1];
            A(i, 2) = v_adj[i][2];
            b(i, 0) = (v_adj[i].norm()) * (v_adj[i].norm()) / 2.0;
        }

        Eigen::Matrix<double, 3, 3> AT_A_inv = (A.transpose() * A).inverse();
        Eigen::Matrix<double, 3, 1> AT_b = A.transpose() * b;
        Eigen::Matrix<double, 3, 1> normal_ = AT_A_inv * AT_b;

        valid = true;
        normal = -(normal_ / normal_.norm());
    }
};

struct Edge
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Edge> Ptr;

    std::vector<std::shared_ptr<Vertex>> vertices;
    int id;
    std::string key;
    std::vector<std::shared_ptr<Triangle>> triangles;
    bool overlap;

    Edge(const std::shared_ptr<Vertex> v1, const std::shared_ptr<Vertex> v2, const int id_)
    {
        vertices = {v1, v2};
        id = id_;

        int v_id_1 = std::min({v1->id, v2->id});
        int v_id_2 = std::max({v1->id, v2->id});

        key = "e_" + std::to_string(v_id_1) + "_" + std::to_string(v_id_2);
        overlap = false;
    }

    void addTriangle(std::shared_ptr<Triangle> triangle)
    {
        triangles.push_back(triangle);
    }
};

struct Triangle
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Triangle> Ptr;

    std::vector<std::shared_ptr<Vertex>> vertices;
    std::shared_ptr<Plane> plane = nullptr;
    int id;
    std::string key;
    std::vector<Edge::Ptr> edges;

    bool overlap;

    bool isSelected;
    int cluster_id;
    bool isSeletedForDepth;

    bool isOnPlane;

    bool valid;

    Triangle(const std::shared_ptr<Vertex> v1, const std::shared_ptr<Vertex> v2, const std::shared_ptr<Vertex> v3, const int id_)
    {
        vertices = {v1, v2, v3};
        id = id_;

        std::vector<int> values = {v1->id, v2->id, v3->id};
        std::vector<size_t> order = argsort_i(values);

        int v_id_1 = vertices[order[0]]->id;
        int v_id_2 = vertices[order[1]]->id;
        int v_id_3 = vertices[order[2]]->id;

        key = "f_" + std::to_string(v_id_1) + "_" + std::to_string(v_id_2) + "_" + std::to_string(v_id_3);
        overlap = false;

        isSelected = false;
        cluster_id = -1;

        valid = false;
        isOnPlane = false;
        isSeletedForDepth = false;
    }

    Eigen::Vector3d evaluateCentroid()
    {
        Eigen::Vector3d sum(vertices[0]->xyz + vertices[1]->xyz + vertices[2]->xyz);
        return sum / 3.0;
    }

    Eigen::Vector3d getCentroid()
    {
        Eigen::Vector3d sum(vertices[0]->xyz + vertices[1]->xyz + vertices[2]->xyz);
        return sum / 3.0;
    }

    Eigen::Vector3d evaluateNormal()
    {
        Eigen::Vector3d AB = (vertices[1]->xyz - vertices[0]->xyz);
        Eigen::Vector3d BC = (vertices[2]->xyz - vertices[1]->xyz);

        return AB.cross(BC) / (AB.cross(BC)).norm();
    }

    Eigen::Vector3d evaluateNormalOrig()
    {
        Eigen::Vector3d AB = (vertices[1]->xyz_orig - vertices[0]->xyz_orig);
        Eigen::Vector3d BC = (vertices[2]->xyz_orig - vertices[1]->xyz_orig);

        return AB.cross(BC) / (AB.cross(BC)).norm();
    }

    double evaluateArea()
    {
        Eigen::Vector3d BA = vertices[0]->xyz - vertices[1]->xyz;
        Eigen::Vector3d BC = vertices[2]->xyz - vertices[1]->xyz;
        double angle = acos(BA.dot(BC) / BA.norm() / BC.norm());
        double triangle_area = BA.norm() * BC.norm() * sin(angle) * 0.5;

        return triangle_area;
    }
};

struct Plane
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Plane> Ptr;

    Eigen::Vector4d coeff;
    int plane_id;
    std::string plane_key;

    Eigen::Vector3d centroid;

    std::vector<std::shared_ptr<Triangle>> triangles;
    std::vector<std::shared_ptr<Vertex>> vertices;

    ///////////////////////////////////////////////////////
    std::vector<std::shared_ptr<Vertex>> minMaxBox;
    bool isMinMaxBoxInitialized;
    Eigen::Vector2f minMaxY;
    Eigen::Vector3d line_horizontal;
    double area; // [m^2]
    double w;
    ///////////////////////////////////////////////////////

    Plane(const Eigen::Vector4d &coeff_, const int plane_id_)
    {
        coeff = coeff_;
        plane_id = plane_id_;
        w = 1.0;
        plane_key = "s_" + std::to_string(plane_id);
        area = 0.0;
        isMinMaxBoxInitialized = false;
    }

    void addTriangle(std::shared_ptr<Triangle> triangle)
    {
        triangle->overlap = false;
        triangle->cluster_id = plane_id;
        triangles.push_back(triangle);
    }

    void updateVertices()
    {
        vertices.clear();

        std::unordered_map<std::string, std::shared_ptr<Vertex>> hash_map_vertex;

        for (auto &triangle : triangles)
        {
            for (auto &vertex : triangle->vertices)
            {

                if (hash_map_vertex.find(vertex->key) == hash_map_vertex.end())
                {
                    hash_map_vertex.insert(std::make_pair(vertex->key, vertex));
                    vertices.push_back(vertex);
                }
            }
        }

        centroid = {0.0, 0.0, 0.0};

        for (auto &vertex : vertices)
            centroid += vertex->xyz;

        centroid /= (double)vertices.size();
    }

    void updateArea()
    {
        area = 0.0;
        for (auto &triangle : triangles)
        {
            Eigen::Vector3d BA = triangle->vertices[0]->xyz - triangle->vertices[1]->xyz;
            Eigen::Vector3d BC = triangle->vertices[2]->xyz - triangle->vertices[1]->xyz;
            double angle = acos(BA.dot(BC) / BA.norm() / BC.norm());
            double triangle_area = BA.norm() * BC.norm() * sin(angle) * 0.5;
            area += triangle_area;
        }
    }
};

// void getLocalTopology(const std::vector<Eigen::Vector3d> &cloud_valid,
//                       const std::vector<std::vector<int>> &triangles,
//                       std::vector<Vertex::Ptr> &vertices_local,
//                       std::vector<Edge::Ptr> &edges_local,
//                       std::vector<Triangle::Ptr> &triangles_local);

// void getLocalTopology(const pcl::PointCloud<PointXYZIT>::Ptr &cloud,
//                       const std::vector<std::vector<int>> &faces,
//                       std::vector<Vertex::Ptr> &vertices_local,
//                       std::vector<Edge::Ptr> &edges_local,
//                       std::vector<Triangle::Ptr> &triangles_local);

void getLocalTopology(const pcl::PointCloud<PointXYZIT>::Ptr &cloud,
                      const std::vector<std::vector<int>> &faces,
                      std::vector<Eigen::Vector3d>& vertices,
                      std::vector<std::vector<std::vector<Eigen::Vector3d>>>& triangles);

void getTriangleMesh(std::vector<Eigen::Vector2f> &uv_valid, std::vector<std::vector<int>> &triangles);

void filterTriangle(const std::vector<Eigen::Vector3d> &cloud,
                    std::vector<std::vector<int>> &triangles,
                    const double distance_threshold,
                    const double distance_ratio_threshold,
                    const double angle_thresh_min,
                    const double angle_thresh_max);

void filterTriangle(const pcl::PointCloud<PointXYZIT>::Ptr &cloud,
                    std::vector<std::vector<int>> &triangles,
                    const double distance_threshold,
                    const double distance_ratio_threshold,
                    const double angle_thresh_min,
                    const double angle_thresh_max);

float sign(const Eigen::Vector2d p1,
           const Eigen::Vector2d p2,
           const Eigen::Vector2d p3);

bool isPointInTriangle(const Eigen::Vector2d pt,
                       const Eigen::Vector2d v1,
                       const Eigen::Vector2d v2,
                       const Eigen::Vector2d v3);

Eigen::Vector3d intersectPointPlaneLine(const Eigen::Vector4d plane,
                                        const Eigen::Vector3d x1,
                                        const Eigen::Vector3d x2);



Eigen::Vector3d evaluateTriangleNormal(std::vector<Eigen::Vector3d> triangle);

double evaluateTriangleArea(std::vector<Eigen::Vector3d> triangle);



#endif