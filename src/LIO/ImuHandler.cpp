#include <ImuHandler.h>

void ImuHandler::resetIntegrator()
{
    imuIntegratorImu_->resetIntegrationAndSetBias(prevBias_);
}

void ImuHandler::integrateIMU(const ImuMessage &imu_msg, Eigen::Matrix4d &Tiw, Eigen::Vector3d &imuVelocity)
{

    double imu_timestamp = imu_msg.timestamp;
    double dt = 0.01;

    if (lastImuTime > 0)
        dt = imu_timestamp - lastImuTime;

    lastImuTime = imu_timestamp;

    Eigen::Vector3d linear_acc = imu_msg.linear_acc - imuMeanAcc;
    Eigen::Vector3d angular_vel = imu_msg.angular_vel;
    imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(linear_acc[0], linear_acc[1], linear_acc[2]), gtsam::Vector3(angular_vel[0], angular_vel[1], angular_vel[2]), dt);

    gtsam::NavState currentState = imuIntegratorImu_->predict(prevStateOdom, prevBiasOdom);
    gtsam::Pose3 imuPose = gtsam::Pose3(currentState.quaternion(), currentState.position());
    Eigen::Vector3d imuVelocity_d = currentState.v();
    imuVelocity = Eigen::Vector3d((double)imuVelocity_d[0], (double)imuVelocity_d[1], (double)imuVelocity_d[2]);

    Eigen::Quaterniond rot_q((double)imuPose.rotation().toQuaternion().w(),
                             (double)imuPose.rotation().toQuaternion().x(),
                             (double)imuPose.rotation().toQuaternion().y(),
                             (double)imuPose.rotation().toQuaternion().z());

    Eigen::Vector3d trans((double)imuPose.translation().x(),
                          (double)imuPose.translation().y(),
                          (double)imuPose.translation().z());

    Eigen::Matrix3d rot = rot_q.normalized().toRotationMatrix();

    Tiw = Eigen::Matrix4d::Identity();
    Tiw.block<3, 3>(0, 0) = rot;
    Tiw.block<3, 1>(0, 3) = trans;
}

void ImuHandler::integrateIMU(const std::vector<ImuMessage> &imu_msg_vec, Eigen::Matrix4d &T_imu_rel, Eigen::Vector3d &imuVelocity)
{
    double last_imu_time = -1;

    for (size_t i = 0; i < imu_msg_vec.size(); i++)
    {
        ImuMessage imu_msg = imu_msg_vec[i];
        double imu_timestamp = imu_msg.timestamp;
        double dt = 0.01;

        if (last_imu_time > 0)
            dt = imu_timestamp - last_imu_time;

        last_imu_time = imu_timestamp;

        Eigen::Vector3d linear_acc = imu_msg.linear_acc - imuMeanAcc;
        Eigen::Vector3d angular_vel = imu_msg.angular_vel;
        imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(linear_acc[0], linear_acc[1], linear_acc[2]), gtsam::Vector3(angular_vel[0], angular_vel[1], angular_vel[2]), dt);
    }

    gtsam::NavState currentState = imuIntegratorImu_->predict(prevStateOdom, prevBiasOdom);
    gtsam::Pose3 imuPose = gtsam::Pose3(currentState.quaternion(), currentState.position());
    Eigen::Vector3d imuVelocity_d = currentState.v();
    imuVelocity = Eigen::Vector3d((double)imuVelocity_d[0], (double)imuVelocity_d[1], (double)imuVelocity_d[2]);

    Eigen::Quaterniond rot_q((double)imuPose.rotation().toQuaternion().w(),
                             (double)imuPose.rotation().toQuaternion().x(),
                             (double)imuPose.rotation().toQuaternion().y(),
                             (double)imuPose.rotation().toQuaternion().z());

    Eigen::Vector3d trans((double)imuPose.translation().x(),
                          (double)imuPose.translation().y(),
                          (double)imuPose.translation().z());

    Eigen::Matrix3d rot = rot_q.normalized().toRotationMatrix();

    T_imu_rel = Eigen::Matrix4d::Identity();
    T_imu_rel.block<3, 3>(0, 0) = rot;
    T_imu_rel.block<3, 1>(0, 3) = trans;

    // imuIntegratorImu_->resetIntegrationAndSetBias(prevBias_);
}

void ImuHandler::optimizeBias(const Eigen::Matrix4d T_wl, const Eigen::Matrix4d T_il)
{
    Eigen::Quaterniond q_wl(T_wl.block<3, 3>(0, 0));
    Eigen::Vector3d p_wl = T_wl.block<3, 1>(0, 3);

    Eigen::Quaterniond q_il(T_il.block<3, 3>(0, 0));
    Eigen::Vector3d p_il = T_il.block<3, 1>(0, 3);

    gtsam::Pose3 lidarPose = gtsam::Pose3(gtsam::Rot3::Quaternion(q_wl.w(), q_wl.x(), q_wl.y(), q_wl.z()), gtsam::Point3(p_wl[0], p_wl[1], p_wl[2]));
    gtsam::Pose3 lidar2Imu = gtsam::Pose3(gtsam::Rot3::Quaternion(q_il.w(), q_il.x(), q_il.y(), q_il.z()), gtsam::Point3(p_il[0], p_il[1], p_il[2]));

    if (!systemInitialized)
    {
        resetOptimization();

        // prior pose (by lidar)
        prevPose_ = lidarPose.compose(lidar2Imu);
        gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, priorPoseNoise);
        graphFactors.add(priorPose);

        // initial velocity
        prevVel_ = gtsam::Vector3(0, 0, 0);
        gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, priorVelNoise);
        graphFactors.add(priorVel);

        // initial bias
        prevBias_ = gtsam::imuBias::ConstantBias();
        gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, priorBiasNoise);
        graphFactors.add(priorBias);

        // add values
        graphValues.insert(X(0), prevPose_);
        graphValues.insert(V(0), prevVel_);
        graphValues.insert(B(0), prevBias_);

        // optimize once
        optimizer.update(graphFactors, graphValues);
        graphFactors.resize(0);
        graphValues.clear();

        imuIntegratorImu_->resetIntegrationAndSetBias(prevBias_);
        imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);

        key = 1;
        systemInitialized = true;

        return;
    }

    // reset graph for speed
    if (key == 30)
    {
        // get updated noise before reset
        gtsam::noiseModel::Gaussian::shared_ptr updatedPoseNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(X(key - 1)));
        gtsam::noiseModel::Gaussian::shared_ptr updatedVelNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(V(key - 1)));
        gtsam::noiseModel::Gaussian::shared_ptr updatedBiasNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(B(key - 1)));
        // reset graph
        resetOptimization();
        // add pose
        gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, updatedPoseNoise);
        graphFactors.add(priorPose);
        // add velocity
        gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, updatedVelNoise);
        graphFactors.add(priorVel);
        // add bias
        gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, updatedBiasNoise);
        graphFactors.add(priorBias);
        // add values
        graphValues.insert(X(0), prevPose_);
        graphValues.insert(V(0), prevVel_);
        graphValues.insert(B(0), prevBias_);
        // optimize once
        optimizer.update(graphFactors, graphValues);
        graphFactors.resize(0);
        graphValues.clear();

        key = 1;
    }

    // 2. Add imu factor to graph
    const gtsam::PreintegratedImuMeasurements &preint_imu = dynamic_cast<const gtsam::PreintegratedImuMeasurements &>(*imuIntegratorImu_);
    gtsam::ImuFactor imu_factor(X(key - 1), V(key - 1), X(key), V(key), B(key - 1), preint_imu);
    graphFactors.add(imu_factor);
    // 3. Add imu bias between factor
    graphFactors.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(B(key - 1), B(key), gtsam::imuBias::ConstantBias(),
                                                                        gtsam::noiseModel::Diagonal::Sigmas(sqrt(imuIntegratorImu_->deltaTij()) * noiseModelBetweenBias)));

    // 4. Add pose factor
    gtsam::Pose3 curPose = lidarPose.compose(lidar2Imu);
    gtsam::PriorFactor<gtsam::Pose3> pose_factor(X(key), curPose, correctionNoise);
    graphFactors.add(pose_factor);

    // 4. Insert predicted values
    gtsam::NavState propState_ = imuIntegratorImu_->predict(prevState_, prevBias_);
    graphValues.insert(X(key), propState_.pose());
    graphValues.insert(V(key), propState_.v());
    graphValues.insert(B(key), prevBias_);

    // 5. Optimize
    optimizer.update(graphFactors, graphValues);
    optimizer.update();
    graphFactors.resize(0);
    graphValues.clear();

    // 6. Overwrite the beginning of the preintegration for the next step.
    gtsam::Values result = optimizer.calculateEstimate();
    prevPose_ = result.at<gtsam::Pose3>(X(key));
    prevVel_ = result.at<gtsam::Vector3>(V(key));
    prevState_ = gtsam::NavState(prevPose_, prevVel_);
    prevBias_ = result.at<gtsam::imuBias::ConstantBias>(B(key));

    // 7. Reset the optimization preintegration object.
    imuIntegratorImu_->resetIntegrationAndSetBias(prevBias_);

    if (failureDetection(prevVel_, prevBias_))
    {
        resetParams();
        return;
    }

    ++key;

    Eigen::Vector3d ba(prevBias_.accelerometer().x(), prevBias_.accelerometer().y(), prevBias_.accelerometer().z());
    Eigen::Vector3d bg(prevBias_.gyroscope().x(), prevBias_.gyroscope().y(), prevBias_.gyroscope().z());

    // std::cout << "Bias Check: " << ba[0] << " // " << ba[1] << " // " << ba[2] << " // " << bg[0] << " // " << bg[1] << " // " << bg[2] << std::endl;
}

bool ImuHandler::failureDetection(const gtsam::Vector3 &velCur, const gtsam::imuBias::ConstantBias &biasCur)
{
    Eigen::Vector3d vel(velCur.x(), velCur.y(), velCur.z());
    if (vel.norm() > 10)
    {
        std::cout << "Large velocity, reset IMU-preintegration!" << std::endl;
        return true;
    }

    Eigen::Vector3d ba(biasCur.accelerometer().x(), biasCur.accelerometer().y(), biasCur.accelerometer().z());
    Eigen::Vector3d bg(biasCur.gyroscope().x(), biasCur.gyroscope().y(), biasCur.gyroscope().z());
    if (ba.norm() > 0.1 || bg.norm() > 0.1)
    {
        std::cout << "Large bias, reset IMU-preintegration!" << std::endl;
        return true;
    }

    return false;
}
