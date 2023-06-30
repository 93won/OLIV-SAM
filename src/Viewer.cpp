#include <Viewer.h>

#include <pangolin/pangolin.h>
#include <ctime>

#include <Config.h>

Viewer::Viewer()
{
}

void Viewer::initialize()
{
    const int UI_WIDTH = 200;
    pangolin::CreateWindowAndBind("GUI", 1024 * 2, 768 * 2);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState vis_camera_(pangolin::ProjectionMatrix(1024, 768, 400, 400, 512, 384, 0.1, 1000),
                                            pangolin::ModelViewLookAt(0, -5, -10, 0, 0, 0, 0.0, -1.0, 0.0));

    vis_camera = vis_camera_;

    vis_display = pangolin::CreateDisplay().SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f).SetHandler(new pangolin::Handler3D(vis_camera));
    pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(UI_WIDTH));
    id = 0;

    cloud_map = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);
    cloud_surf = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);
    cloud_map_local = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);
    cloud_feature = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);
    cloud_cumulated = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);

    // std::random_device rd;
    // std::mt19937 gen(rd());
    // std::uniform_real_distribution<> dis(0, 1);

    // for (int i = 0; i < 100000; i++)
    // {
    //     std::vector<double> color = {dis(gen), dis(gen), dis(gen)};
    //     colors.push_back(color);
    // }
}

void Viewer::drawPoints(std::vector<Eigen::Vector3d> &points, std::vector<Eigen::Vector3d> &colors, int size = 4)
{
    glPointSize(size);
    glBegin(GL_POINTS);
    for (size_t i = 0; i < points.size(); i++)
    {
        glColor3f(colors[i][0], colors[i][1], colors[i][2]);
        glVertex3d(points[i][0], points[i][1], points[i][2]);
    }
    glEnd();
}

void Viewer::drawPoints(pcl::PointCloud<PointType>::Ptr &cloud, Eigen::Vector3d &colors, int size = 4)
{
    glPointSize(size);
    glBegin(GL_POINTS);
    for (size_t i = 0; i < cloud->points.size(); i++)
    {
        glColor3f(colors[0], colors[1], colors[2]);
        glVertex3d(cloud->points[i].x, cloud->points[i].y, cloud->points[i].z);
    }
    glEnd();
}

void Viewer::drawKeyFrame(Frame::Ptr frame)
{
    Eigen::Matrix4d Twc = frame->Pose().inverse();
    const float sz = 0.3;
    const int line_width = 2.0;

    glPushMatrix();

    Eigen::Matrix4f m = Twc.matrix().template cast<float>();
    glMultMatrixf((GLfloat *)m.data());

    glColor3f(1,0,0);

    glLineWidth(line_width);
    glBegin(GL_LINES);
    glVertex3f(0, 0, 0);
    glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
    glVertex3f(0, 0, 0);
    glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
    glVertex3f(0, 0, 0);
    glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
    glVertex3f(0, 0, 0);
    glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);

    glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
    glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);

    glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
    glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);

    glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
    glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);

    glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
    glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);

    glEnd();
    glPopMatrix();
}

void Viewer::drawFrame(Frame::Ptr frame)
{
    Eigen::Matrix4d Twc = frame->Pose().inverse();
    const float sz = 0.3;
    const int line_width = 2.0;

    glPushMatrix();

    Eigen::Matrix4f m = Twc.matrix().template cast<float>();
    glMultMatrixf((GLfloat *)m.data());

    glColor3f(0,0,1);

    glLineWidth(line_width);
    glBegin(GL_LINES);
    glVertex3f(0, 0, 0);
    glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
    glVertex3f(0, 0, 0);
    glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
    glVertex3f(0, 0, 0);
    glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
    glVertex3f(0, 0, 0);
    glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);

    glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
    glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);

    glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
    glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);

    glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
    glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);

    glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
    glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);

    glEnd();
    glPopMatrix();
}

void Viewer::drawAxis(Eigen::Matrix4d pose)
{
    Eigen::Matrix3d R = pose.block<3, 3>(0, 0);
    Eigen::Vector3d t = pose.block<3, 1>(0, 3);

    double ratio = 0.5;

    Eigen::Vector3d dir_x = R * Eigen::Vector3d(1.0, 0.0, 0.0) * ratio;
    Eigen::Vector3d dir_y = R * Eigen::Vector3d(0.0, 1.0, 0.0) * ratio;
    Eigen::Vector3d dir_z = R * Eigen::Vector3d(0.0, 0.0, 1.0) * ratio;

    glLineWidth(6);
    glBegin(GL_LINES);
    glColor3f(1.0, 0.0, 0.0);
    glVertex3d(t[0], t[1], t[2]);
    glVertex3d(t[0] + dir_x[0], t[1] + dir_x[1], t[2] + dir_x[2]);
    glColor3f(0.0, 1.0, 0.0);
    glVertex3d(t[0], t[1], t[2]);
    glVertex3d(t[0] + dir_y[0], t[1] + dir_y[1], t[2] + dir_y[2]);
    glColor3f(0.0, 0.0, 1.0);
    glVertex3d(t[0], t[1], t[2]);
    glVertex3d(t[0] + dir_z[0], t[1] + dir_z[1], t[2] + dir_z[2]);
    glEnd();

    glPointSize(10);
    glBegin(GL_POINTS);
    glColor3f(0.0, 0.0, 0.0);
    glVertex3d(t[0], t[1], t[2]);
    glEnd();
}

void Viewer::drawAxis(Eigen::Matrix4d pose, Eigen::Vector3d &colors)
{
    Eigen::Matrix3d R = pose.block<3, 3>(0, 0);
    Eigen::Vector3d t = pose.block<3, 1>(0, 3);

    double ratio = 0.5;

    Eigen::Vector3d dir_x = R * Eigen::Vector3d(1.0, 0.0, 0.0) * ratio;
    Eigen::Vector3d dir_y = R * Eigen::Vector3d(0.0, 1.0, 0.0) * ratio;
    Eigen::Vector3d dir_z = R * Eigen::Vector3d(0.0, 0.0, 1.0) * ratio;

    glColor3f(colors[0], colors[1], colors[2]);
    glLineWidth(6);
    glBegin(GL_LINES);
    glVertex3d(t[0], t[1], t[2]);
    glVertex3d(t[0] + dir_x[0], t[1] + dir_x[1], t[2] + dir_x[2]);
    glVertex3d(t[0], t[1], t[2]);
    glVertex3d(t[0] + dir_y[0], t[1] + dir_y[1], t[2] + dir_y[2]);
    glVertex3d(t[0], t[1], t[2]);
    glVertex3d(t[0] + dir_z[0], t[1] + dir_z[1], t[2] + dir_z[2]);
    glEnd();

    glPointSize(10);
    glBegin(GL_POINTS);
    glColor3f(0.0, 0.0, 0.0);
    glVertex3d(t[0], t[1], t[2]);
    glEnd();
}

void Viewer::drawAxisThin(Eigen::Matrix4d pose, Eigen::Vector3d &colors)
{
    Eigen::Matrix3d R = pose.block<3, 3>(0, 0);
    Eigen::Vector3d t = pose.block<3, 1>(0, 3);

    double ratio = 0.8;

    Eigen::Vector3d dir_x = R * Eigen::Vector3d(1.0, 0.0, 0.0) * ratio;
    Eigen::Vector3d dir_y = R * Eigen::Vector3d(0.0, 1.0, 0.0) * ratio;
    Eigen::Vector3d dir_z = R * Eigen::Vector3d(0.0, 0.0, 1.0) * ratio;

    glColor3f(colors[0], colors[1], colors[2]);
    glLineWidth(3);
    glBegin(GL_LINES);
    glVertex3d(t[0], t[1], t[2]);
    glVertex3d(t[0] + dir_x[0], t[1] + dir_x[1], t[2] + dir_x[2]);
    glVertex3d(t[0], t[1], t[2]);
    glVertex3d(t[0] + dir_y[0], t[1] + dir_y[1], t[2] + dir_y[2]);
    glVertex3d(t[0], t[1], t[2]);
    glVertex3d(t[0] + dir_z[0], t[1] + dir_z[1], t[2] + dir_z[2]);
    glEnd();

    glPointSize(10);
    glBegin(GL_POINTS);
    glColor3f(0.0, 0.0, 0.0);
    glVertex3d(t[0], t[1], t[2]);
    glEnd();
}
void Viewer::drawTrajectory(Eigen::Vector3d color, std::vector<Eigen::Matrix4d> trajectory, int size = 4)
{

    // glPointSize(size);
    glBegin(GL_LINES);
    glColor3f(color[0], color[1], color[2]);

    std::vector<Eigen::Vector3d> xyz_trajectory;

    for (auto &pose : trajectory)
    {

        Eigen::Matrix3d R = pose.block<3, 3>(0, 0);
        Eigen::Vector3d trans = pose.block<3, 1>(0, 3);
        xyz_trajectory.push_back(trans);
        // std::cout << "DRAW AXIS: " << trans.transpose() << std::endl;
    }

    for (size_t i = 0; i < xyz_trajectory.size() - 1; i++)
    {
        glVertex3d(xyz_trajectory[i][0], xyz_trajectory[i][1], xyz_trajectory[i][2]);
        glVertex3d(xyz_trajectory[i + 1][0], xyz_trajectory[i + 1][1], xyz_trajectory[i + 1][2]);
    }

    glEnd();
}

void Viewer::drawTriangles(const std::vector<Triangle::Ptr> &triangles)
{

    for (auto &triangle : triangles)
    {

        if (!triangle->isSeletedForDepth)
        {
            glLineWidth(2);
            glColor3f(0.0f, 0.0f, 0.0f);
        }

        else
        {
            glColor3f(1.0f, 0.0f, 0.0f);
            glLineWidth(10);
        }

        glBegin(GL_LINES);
        // glColor3f(0.0f, 0.0f, 0.0f);

        // center of inner circle
        Eigen::Vector3d A = triangle->vertices[0]->xyz; // worldToCamera(T_cw, K, triangle->vertices[0]->xyz);
        Eigen::Vector3d B = triangle->vertices[1]->xyz; // worldToCamera(T_cw, K, triangle->vertices[1]->xyz);
        Eigen::Vector3d C = triangle->vertices[2]->xyz; // worldToCamera(T_cw, K, triangle->vertices[2]->xyz);

        glVertex3d(A[0], A[1], A[2]);
        glVertex3d(B[0], B[1], B[2]);

        glVertex3d(B[0], B[1], B[2]);
        glVertex3d(C[0], C[1], C[2]);

        glVertex3d(C[0], C[1], C[2]);
        glVertex3d(A[0], A[1], A[2]);

        glEnd();
    }
}

void Viewer::spinOnce()
{

    // Buttons
    pangolin::Var<bool> button_up_id = pangolin::Var<bool>("ui.Up_id", false, false);
    pangolin::Var<bool> button_point_cloud = pangolin::Var<bool>("ui.Show_points", false, false);
    pangolin::Var<bool> button_mesh = pangolin::Var<bool>("ui.Show_mesh", false, false);
    pangolin::Var<bool> button_start = pangolin::Var<bool>("ui.Run", false, false);

    if (pangolin::Pushed(button_mesh))
    {
        if (!flagDrawMesh)
        {
            flagDrawMesh = true;
            std::cout << "Draw Current Mesh" << std::endl;
        }

        else
        {
            flagDrawMesh = false;
            std::cout << "Hide Current Mesh" << std::endl;
        }
    }

    if (pangolin::Pushed(button_point_cloud))
    {
        if (!flagDrawCurrentPointCloud)
        {
            flagDrawCurrentPointCloud = true;
            std::cout << "Draw Points" << std::endl;
        }

        else
        {
            flagDrawCurrentPointCloud = false;
            std::cout << "Hide Points" << std::endl;
        }
    }

    if (pangolin::Pushed(button_start))
    {
        if (!start)
            start = true;

        else
            start = false;
    }

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    vis_display.Activate(vis_camera);

    const double green[3] = {0.0, 1.0, 0.0};
    const double black[3] = {0.0, 0.0, 0.0};

    Eigen::Vector3d color_red(1.0, 0.0, 0.0);
    Eigen::Vector3d color_green(0.0, 1.0, 0.0);
    Eigen::Vector3d color_blue(0.0, 0.0, 1.0);
    Eigen::Vector3d color_black(0.0, 0.0, 0.0);

    std::unique_lock<std::mutex> lock(viewer_data_mutex_);

    const double red[3] = {1.0, 0, 0};

    if (pangolin::Pushed(button_up_id))
        id += 1;

    if (flagDrawCurrentPointCloud)
    {
        // drawPoints(cloud_surf, color_blue, 8);
        drawPoints(cloud_map_local, color_red, 4);
        drawPoints(cloud_map, color_black, 2);
        drawPoints(cloud_feature, color_blue, 8);
    }

    // drawTrajectory(Eigen::Vector3d(1.0, 0.0, 0.0), trajectory);
    // drawTrajectory(Eigen::Vector3d(0.0, 0.0, 0.0), trajectory_imu, 4);

    if (flagDrawMesh)
        drawTriangles(local_triangles);

    // drawAxis(T_wl);
    // drawAxis(T_wc);

    if(keyframes.size() > 0)
    {
        for(auto& frame : keyframes)
            drawKeyFrame(frame);
    }
    if(curr_frame != nullptr)
        drawFrame(curr_frame);

    pangolin::FinishFrame();
}
