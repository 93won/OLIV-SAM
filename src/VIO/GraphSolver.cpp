#include <GraphSolver.h>

// void GraphSolver::initialize(double timestamp, Eigen::Matrix4d Tcw)
// {
//     if (systeminitalized)
//         return;

//     gtsam::Pose3 initialPose = gtsam::Pose3(gtsam::Rot3::Quaternion(q_wl.w(), q_wl.x(), q_wl.y(), q_wl.z()), gtsam::Point3(p_wl[0], p_wl[1], p_wl[2]));
//     gtsam::Pose3 lidar2Imu = gtsam::Pose3(gtsam::Rot3::Quaternion(q_il.w(), q_il.x(), q_il.y(), q_il.z()), gtsam::Point3(p_il[0], p_il[1], p_il[2]));

    

    
// }

gtsam::CombinedImuFactor GraphSolver::createImuFactor(const std::vector<ImuMessage> &imuMsgVec)
{
    // Assume imu msgs in imuMsgVec are imu measurement between two consecutive frames

    double dt = 0.01;
    for (size_t i = 0; i < imuMsgVec.size() - 1; i++)
    {
        dt = imuMsgVec[i + 1].timestamp - imuMsgVec[i].timestamp;
        preintGTSAM->integrateMeasurement(imuMsgVec[i].linear_acc, imuMsgVec[i].angular_vel, dt);
    }
    preintGTSAM->integrateMeasurement(imuMsgVec[imuMsgVec.size()].linear_acc, imuMsgVec[imuMsgVec.size()].angular_vel, dt);

    return gtsam::CombinedImuFactor(X(key), V(key), X(key + 1), V(key + 1), B(key), B(key + 1), *preintGTSAM);
}