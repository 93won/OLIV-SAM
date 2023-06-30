#include <GeometryUtils.h>

Eigen::Vector3d evaluateTriangleNormal(std::vector<Eigen::Vector3d> triangle)
{
    Eigen::Vector3d AB = (triangle[1] - triangle[0]);
    Eigen::Vector3d BC = (triangle[2] - triangle[1]);

    return AB.cross(BC) / (AB.cross(BC)).norm();
}

double evaluateTriangleArea(std::vector<Eigen::Vector3d> triangle)
{
    Eigen::Vector3d BA = triangle[0] - triangle[1];
    Eigen::Vector3d BC = triangle[2] - triangle[1];
    double angle = acos(BA.dot(BC) / BA.norm() / BC.norm());
    double triangle_area = BA.norm() * BC.norm() * sin(angle) * 0.5;

    return triangle_area;
}

void getTriangleMesh(std::vector<Eigen::Vector2f> &uv_valid, std::vector<std::vector<int>> &triangles)
{
    float u_min = 1000.0;
    float v_min = 1000.0;
    float u_max = -1000.0;
    float v_max = -1000.0;

    for (auto &uv : uv_valid)
    {
        if (uv[0] < u_min)
            u_min = uv[0];

        if (uv[1] < v_min)
            v_min = uv[1];

        if (uv[0] > u_max)
            u_max = uv[0];

        if (uv[1] > v_max)
            v_max = uv[1];
    }

    for (auto &uv : uv_valid)
    {
        uv[0] -= u_min;
        uv[1] -= v_min;
    }

    u_min -= 1;
    v_min -= 1;

    u_max -= u_min;
    v_max -= v_min;


    // for (auto &uv : uv_valid)
    // {
    //     if (uv[0] < 0 || uv[0] > u_max)
    //         std::cout << "E1" << std::endl;

    //     if (uv[1] < 0 || uv[1] > v_max)
    //         std::cout << "E2" << std::endl;

    //     std::cout << "uv: " << uv[0] << " " << uv[1] << std::endl;
    // }

    cv::Rect ROI = cv::Rect(0, 0, v_max, u_max);
    // cv::Mat img = cv::imread(img_path_full)(ROI);

    Subdiv2DIndex subdiv(ROI);

    for (size_t i = 0; i < uv_valid.size(); ++i)
    {
        cv::Point2f fp(uv_valid[i][1], uv_valid[i][0]);
        subdiv.insert(fp);
    }

    subdiv.getTrianglesIndices(triangles);

    // std::cout<<"# of triangles: "<<triangles.size()<<std::endl;
}

void filterTriangle(const pcl::PointCloud<PointXYZIT>::Ptr &cloud,
                    std::vector<std::vector<int>> &triangles,
                    const double distance_threshold,
                    const double distance_ratio_threshold,
                    const double angle_thresh_min,
                    const double angle_thresh_max)
{
    std::vector<std::vector<int>> triangles_filtered;

    int nb_triangles = (int)triangles.size();
    for (int i = 0; i < nb_triangles; i++)
    {
        Eigen::Vector3d p_A = {cloud->points[triangles[i][0]].x, cloud->points[triangles[i][0]].y, cloud->points[triangles[i][0]].z};
        Eigen::Vector3d p_B = {cloud->points[triangles[i][1]].x, cloud->points[triangles[i][1]].y, cloud->points[triangles[i][1]].z};
        Eigen::Vector3d p_C = {cloud->points[triangles[i][2]].x, cloud->points[triangles[i][2]].y, cloud->points[triangles[i][2]].z};

        double angle_A = acos((p_B - p_A).dot(p_C - p_A) / (p_B - p_A).norm() / (p_C - p_A).norm());
        double angle_B = acos((p_A - p_B).dot(p_C - p_B) / (p_A - p_B).norm() / (p_C - p_B).norm());
        double angle_C = acos((p_A - p_C).dot(p_B - p_C) / (p_A - p_C).norm() / (p_B - p_C).norm());

        if (angle_A > angle_thresh_max || angle_B > angle_thresh_max || angle_C > angle_thresh_max)
            continue;

        if (angle_A < angle_thresh_min || angle_B < angle_thresh_min || angle_C < angle_thresh_min)
            continue;

        double distAB = (p_B - p_A).norm();
        double distBC = (p_C - p_B).norm();
        double distCA = (p_A - p_C).norm();

        double max = std::max({distAB, distBC, distCA});
        double min = std::min({distAB, distBC, distCA});

        double ratio = min / max;

        bool isValidTriangle = distAB < distance_threshold &&
                               distBC < distance_threshold &&
                               distCA < distance_threshold &&
                               ratio > distance_ratio_threshold;

        if (isValidTriangle)
            triangles_filtered.push_back(triangles[i]);
    }

    triangles = triangles_filtered;
}

void filterTriangle(const std::vector<Eigen::Vector3d> &cloud,
                    std::vector<std::vector<int>> &triangles,
                    const double distance_threshold,
                    const double distance_ratio_threshold,
                    const double angle_thresh_min,
                    const double angle_thresh_max)
{

    std::vector<std::vector<int>> triangles_filtered;

    int nb_triangles = (int)triangles.size();
    for (int i = 0; i < nb_triangles; i++)
    {
        Eigen::Vector3d p_A = {cloud[triangles[i][0]][0], cloud[triangles[i][0]][1], cloud[triangles[i][0]][2]};
        Eigen::Vector3d p_B = {cloud[triangles[i][1]][0], cloud[triangles[i][1]][1], cloud[triangles[i][1]][2]};
        Eigen::Vector3d p_C = {cloud[triangles[i][2]][0], cloud[triangles[i][2]][1], cloud[triangles[i][2]][2]};

        double angle_A = acos((p_B - p_A).dot(p_C - p_A) / (p_B - p_A).norm() / (p_C - p_A).norm());
        double angle_B = acos((p_A - p_B).dot(p_C - p_B) / (p_A - p_B).norm() / (p_C - p_B).norm());
        double angle_C = acos((p_A - p_C).dot(p_B - p_C) / (p_A - p_C).norm() / (p_B - p_C).norm());

        if (angle_A > angle_thresh_max || angle_B > angle_thresh_max || angle_C > angle_thresh_max)
            continue;

        if (angle_A < angle_thresh_min || angle_B < angle_thresh_min || angle_C < angle_thresh_min)
            continue;

        double distAB = (p_B - p_A).norm();
        double distBC = (p_C - p_B).norm();
        double distCA = (p_A - p_C).norm();

        double max = std::max({distAB, distBC, distCA});
        double min = std::min({distAB, distBC, distCA});

        double ratio = min / max;

        bool isValidTriangle = distAB < distance_threshold &&
                               distBC < distance_threshold &&
                               distCA < distance_threshold &&
                               ratio > distance_ratio_threshold;

        if (isValidTriangle)
            triangles_filtered.push_back(triangles[i]);
    }

    triangles = triangles_filtered;
}

void getLocalTopology(const pcl::PointCloud<PointXYZIT>::Ptr &cloud,
                      const std::vector<std::vector<int>> &faces,
                      std::vector<Eigen::Vector3d> &vertices,
                      std::vector<std::vector<std::vector<Eigen::Vector3d>>> &triangles)
{

    // vertex 별로 붙어있는 triangle을 vector 형태로 저장하면 어때?
    vertices.resize(cloud->points.size());
    triangles.resize(cloud->points.size());

    for (auto &tri : faces)
    {
        std::vector<Eigen::Vector3d> triangle(3);
        for (int i = 0; i < 3; i++)
        {
            Eigen::Vector3d xyz = {cloud->points[tri[i]].x, cloud->points[tri[i]].y, cloud->points[tri[i]].z};
            vertices[tri[i]] = xyz; // not a big deal
            triangle[i] = xyz;
        }

        for (int i = 0; i < 3; i++)
            triangles[tri[i]].push_back(triangle);
    }

    // std::unordered_map<std::string, Vertex::Ptr> hash_map_vertex_local;
    // std::unordered_map<std::string, Edge::Ptr> hash_map_edge_local;
    // int NB_VERTICES = 0;
    // int NB_EDGES = 0;
    // int NB_TRIANGLES = 0;

    // // local key: to prevent duplication
    // // add triangle to edge (like winged edge)
    // for (auto &tri : faces)
    // {
    //     std::vector<Vertex::Ptr> vertex_triangle;

    //     for (int i = 0; i < 3; i++)
    //     {
    //         std::string key_v_i = "v_" + std::to_string(tri[i]);
    //         if (hash_map_vertex_local.find(key_v_i) == hash_map_vertex_local.end())
    //         {
    //             Eigen::Vector3d xyz = {cloud->points[tri[i]].x, cloud->points[tri[i]].y, cloud->points[tri[i]].z};
    //             Vertex::Ptr vertex_i(new Vertex(xyz, NB_VERTICES++, cloud->points[tri[i]].offset_time, cloud->points[tri[i]].intensity));

    //             vertices_local.push_back(vertex_i);
    //             hash_map_vertex_local.insert(std::make_pair(key_v_i, vertex_i));
    //             vertex_triangle.push_back(vertex_i);
    //         }
    //         else
    //         {
    //             Vertex::Ptr vertex_i = (hash_map_vertex_local.find(key_v_i))->second;
    //             vertex_triangle.push_back(vertex_i);
    //         }
    //     }

    //     // check counter-clockwise
    //     Vertex::Ptr A, B, C;
    //     Eigen::Vector3d vec1 = (vertex_triangle[1]->xyz - vertex_triangle[0]->xyz);
    //     Eigen::Vector3d vec2 = (vertex_triangle[2]->xyz - vertex_triangle[1]->xyz);
    //     Eigen::Vector3d normal = vec1.cross(vec2);
    //     Eigen::Vector3d o2c = (vertex_triangle[0]->xyz + vertex_triangle[1]->xyz + vertex_triangle[2]->xyz) / 3;

    //     if (normal.dot(o2c) < 0)
    //     {
    //         A = vertex_triangle[0];
    //         B = vertex_triangle[1];
    //         C = vertex_triangle[2];
    //     }

    //     else
    //     {
    //         A = vertex_triangle[0];
    //         B = vertex_triangle[2];
    //         C = vertex_triangle[1];
    //     }

    //     Triangle::Ptr triangle(new Triangle(A, B, C, NB_TRIANGLES++));

    //     triangles_local.push_back(triangle);

    //     vertex_triangle[0]->addTriangle(triangle, triangle->key);
    //     vertex_triangle[1]->addTriangle(triangle, triangle->key);
    //     vertex_triangle[2]->addTriangle(triangle, triangle->key);

    //     for (int i = 0; i < 3; i++)
    //     {
    //         int v_id_1 = std::min({tri[i], tri[(i + 1) % 3]});
    //         int v_id_2 = std::max({tri[i], tri[(i + 1) % 3]});

    //         std::string key_e_i = "e_" + std::to_string(v_id_1) + "_" + std::to_string(v_id_2);

    //         if (hash_map_edge_local.find(key_e_i) == hash_map_edge_local.end())
    //         {
    //             // Edge(const std::shared_ptr<Vertex> v1, const std::shared_ptr<Vertex> v2, const int id_)
    //             std::string key_v_1 = "v_" + std::to_string(v_id_1);
    //             std::string key_v_2 = "v_" + std::to_string(v_id_2);

    //             Vertex::Ptr v_1 = (hash_map_vertex_local.find(key_v_1))->second;
    //             Vertex::Ptr v_2 = (hash_map_vertex_local.find(key_v_2))->second;

    //             v_1->vertices_adj.push_back(v_2);
    //             v_2->vertices_adj.push_back(v_1);

    //             Edge::Ptr edge_i(new Edge(v_1, v_2, NB_EDGES++));
    //             edge_i->addTriangle(triangle);
    //             triangle->edges.push_back(edge_i);
    //             hash_map_edge_local.insert(std::make_pair(key_e_i, edge_i));
    //             edges_local.push_back(edge_i);
    //         }

    //         else
    //         {
    //             Edge::Ptr edge_i = (hash_map_edge_local.find(key_e_i))->second;
    //             edge_i->addTriangle(triangle);
    //             triangle->edges.push_back(edge_i);
    //         }
    //     }
    // }

    // hash_map_edge_local.clear();
    // hash_map_vertex_local.clear();
}

// void getLocalTopology(const std::vector<Eigen::Vector3d> &cloud_valid,
//                       const std::vector<std::vector<int>> &triangles,
//                       std::vector<Vertex::Ptr> &vertices_local,
//                       std::vector<Edge::Ptr> &edges_local,
//                       std::vector<Triangle::Ptr> &triangles_local)
// {
//     std::unordered_map<std::string, Vertex::Ptr> hash_map_vertex_local;
//     std::unordered_map<std::string, Edge::Ptr> hash_map_edge_local;
//     int NB_VERTICES = 0;
//     int NB_EDGES = 0;
//     int NB_TRIANGLES = 0;

//     // local key: to prevent duplication
//     // add triangle to edge (like winged edge)
//     for (auto &tri : triangles)
//     {
//         std::vector<Vertex::Ptr> vertex_triangle;

//         for (int i = 0; i < 3; i++)
//         {
//             std::string key_v_i = "v_" + std::to_string(tri[i]);
//             if (hash_map_vertex_local.find(key_v_i) == hash_map_vertex_local.end())
//             {
//                 Eigen::Vector3d xyz = {cloud_valid[tri[i]][0], cloud_valid[tri[i]][1], cloud_valid[tri[i]][2]};
//                 Vertex::Ptr vertex_i(new Vertex(xyz, NB_VERTICES++));

//                 vertices_local.push_back(vertex_i);
//                 hash_map_vertex_local.insert(std::make_pair(key_v_i, vertex_i));
//                 vertex_triangle.push_back(vertex_i);
//             }
//             else
//             {
//                 Vertex::Ptr vertex_i = (hash_map_vertex_local.find(key_v_i))->second;
//                 vertex_triangle.push_back(vertex_i);
//             }
//         }

//         // check counter-clockwise
//         Vertex::Ptr A, B, C;
//         Eigen::Vector3d vec1 = (vertex_triangle[1]->xyz - vertex_triangle[0]->xyz);
//         Eigen::Vector3d vec2 = (vertex_triangle[2]->xyz - vertex_triangle[1]->xyz);
//         Eigen::Vector3d normal = vec1.cross(vec2);
//         Eigen::Vector3d o2c = (vertex_triangle[0]->xyz + vertex_triangle[1]->xyz + vertex_triangle[2]->xyz) / 3;

//         if (normal.dot(o2c) < 0)
//         {
//             A = vertex_triangle[0];
//             B = vertex_triangle[1];
//             C = vertex_triangle[2];
//         }

//         else
//         {
//             A = vertex_triangle[0];
//             B = vertex_triangle[2];
//             C = vertex_triangle[1];
//         }

//         Triangle::Ptr triangle(new Triangle(A, B, C, NB_TRIANGLES++));

//         triangles_local.push_back(triangle);

//         vertex_triangle[0]->addTriangle(triangle, triangle->key);
//         vertex_triangle[1]->addTriangle(triangle, triangle->key);
//         vertex_triangle[2]->addTriangle(triangle, triangle->key);

//         for (int i = 0; i < 3; i++)
//         {
//             int v_id_1 = std::min({tri[i], tri[(i + 1) % 3]});
//             int v_id_2 = std::max({tri[i], tri[(i + 1) % 3]});

//             std::string key_e_i = "e_" + std::to_string(v_id_1) + "_" + std::to_string(v_id_2);

//             if (hash_map_edge_local.find(key_e_i) == hash_map_edge_local.end())
//             {
//                 // Edge(const std::shared_ptr<Vertex> v1, const std::shared_ptr<Vertex> v2, const int id_)
//                 std::string key_v_1 = "v_" + std::to_string(v_id_1);
//                 std::string key_v_2 = "v_" + std::to_string(v_id_2);

//                 Vertex::Ptr v_1 = (hash_map_vertex_local.find(key_v_1))->second;
//                 Vertex::Ptr v_2 = (hash_map_vertex_local.find(key_v_2))->second;

//                 v_1->vertices_adj.push_back(v_2);
//                 v_2->vertices_adj.push_back(v_1);

//                 Edge::Ptr edge_i(new Edge(v_1, v_2, NB_EDGES++));
//                 edge_i->addTriangle(triangle);
//                 triangle->edges.push_back(edge_i);
//                 hash_map_edge_local.insert(std::make_pair(key_e_i, edge_i));
//                 edges_local.push_back(edge_i);
//             }

//             else
//             {
//                 Edge::Ptr edge_i = (hash_map_edge_local.find(key_e_i))->second;
//                 edge_i->addTriangle(triangle);
//                 triangle->edges.push_back(edge_i);
//             }
//         }
//     }

//     hash_map_edge_local.clear();
//     hash_map_vertex_local.clear();
// }

float sign(const Eigen::Vector2d p1,
           const Eigen::Vector2d p2,
           const Eigen::Vector2d p3)
{
    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1]);
}

bool isPointInTriangle(const Eigen::Vector2d pt,
                       const Eigen::Vector2d v1,
                       const Eigen::Vector2d v2,
                       const Eigen::Vector2d v3)
{
    float d1, d2, d3;
    bool has_neg, has_pos;

    d1 = sign(pt, v1, v2);
    d2 = sign(pt, v2, v3);
    d3 = sign(pt, v3, v1);

    has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
    has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);

    return !(has_neg && has_pos);
}

Eigen::Vector3d intersectPointPlaneLine(const Eigen::Vector4d plane,
                                        const Eigen::Vector3d x1,
                                        const Eigen::Vector3d x2)
{
    Eigen::Vector3d d = x2 - x1;
    Eigen::Vector3d n(plane[0], plane[1], plane[2]);

    double t = -(n.dot(x1) + plane[3]) / (n.dot(d));
    return Eigen::Vector3d(d[0] * t + x1[0], d[1] * t + x1[1], d[2] * t + x1[2]);
}