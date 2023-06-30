#pragma once
#ifndef IMU_HANDLER_H
#define IMU_HANDLER_H

#include <Common.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>

using gtsam::symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)
using gtsam::symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using gtsam::symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)

class ImuHandler
{
public:
    Eigen::Matrix4d T_il;

    // IMU
    double imuAccNoise;
    double imuGyrNoise;
    double imuAccBiasN;
    double imuGyrBiasN;
    double imuGravity;

    Eigen::Vector3d imuMeanAcc;

    std::vector<double> extRotV;
    std::vector<double> extRPYV;
    std::vector<double> extTransV;
    Eigen::Matrix3d extRot;
    Eigen::Matrix3d extRPY;
    Eigen::Vector3d extTrans;
    Eigen::Quaterniond extQRPY;

    bool systemInitialized = false;

    gtsam::noiseModel::Diagonal::shared_ptr priorPoseNoise;
    gtsam::noiseModel::Diagonal::shared_ptr priorVelNoise;
    gtsam::noiseModel::Diagonal::shared_ptr priorBiasNoise;
    gtsam::noiseModel::Diagonal::shared_ptr correctionNoise;
    //correctionNoise = gtsam::noiseModel::Isotropic::Sigma(6, 1e-2);  
    gtsam::Vector noiseModelBetweenBias;

    gtsam::PreintegratedImuMeasurements *imuIntegratorOpt_;
    gtsam::PreintegratedImuMeasurements *imuIntegratorImu_;

    // std::deque<sensor_msgs::Imu> imuQueOpt;
    // std::deque<sensor_msgs::Imu> imuQueImu;

    gtsam::Pose3 prevPose_;
    gtsam::Vector3 prevVel_;
    gtsam::NavState prevState_;
    gtsam::imuBias::ConstantBias prevBias_;

    gtsam::NavState prevStateOdom;
    gtsam::imuBias::ConstantBias prevBiasOdom;


    // bool doneFirstOpt = false;
    // double lastImuT_imu = -1;
    // double lastImuT_opt = -1;

    gtsam::ISAM2 optimizer;
    gtsam::NonlinearFactorGraph graphFactors;
    gtsam::Values graphValues;

    double lastImuTime;
    int key;

    std::deque<ImuMessage> imu_queue;
    bool imuInit = false;

    ///////////////////////////////////////////////

    Eigen::Matrix4d T_wi_curr;
    Eigen::Matrix4d T_wi_prev;

    ImuHandler()
    {
        T_il << -0.99999353, 0.00169655, 0.00327721, -0.0554467,
            0.00172166, 0.99996872, 0.00767292, 0.00517844,
            -0.00326409, 0.0076785, -0.99996512, 0.03128449,
            0, 0, 0, 1;

        imuAccNoise = 0.0019139261498958794;
        imuGyrNoise = 0.0007702111507148731;
        imuAccBiasN = 3.1498719039718536e-05;
        imuGyrBiasN = 3.116143253015099e-05;

        imuMeanAcc = Eigen::Vector3d(0, 0, 0);
        T_wi_curr = Eigen::Matrix4d::Identity();
        T_wi_prev = Eigen::Matrix4d::Identity();

        boost::shared_ptr<gtsam::PreintegrationParams> p = gtsam::PreintegrationParams::MakeSharedU(0.0);

        p->accelerometerCovariance = gtsam::Matrix33::Identity(3, 3) * pow(imuAccNoise, 2);             // acc white noise in continuous
        p->gyroscopeCovariance = gtsam::Matrix33::Identity(3, 3) * pow(imuGyrNoise, 2);                 // gyro white noise in continuous
        p->integrationCovariance = gtsam::Matrix33::Identity(3, 3) * pow(1e-4, 2);                      // error committed in integrating position from velocities
        gtsam::imuBias::ConstantBias prior_imu_bias((gtsam::Vector(6) << 0, 0, 0, 0, 0, 0).finished()); // assume zero initial bias

        prevBias_ = gtsam::imuBias::ConstantBias((gtsam::Vector(6) << 0, 0, 0, 0, 0, 0).finished());
        priorPoseNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2).finished()); // rad,rad,rad,m, m, m
        priorVelNoise = gtsam::noiseModel::Isotropic::Sigma(3, 1e2);                                                               // m/s
        priorBiasNoise = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3);                                                             // 1e-2 ~ 1e-3 seems to be good
        correctionNoise = gtsam::noiseModel::Isotropic::Sigma(6, 1e-2);                                                            // meter
        noiseModelBetweenBias = (gtsam::Vector(6) << imuAccBiasN, imuAccBiasN, imuAccBiasN, imuGyrBiasN, imuGyrBiasN, imuGyrBiasN).finished();

        imuIntegratorImu_ = new gtsam::PreintegratedImuMeasurements(p, prevBias_); // setting up the IMU integration for IMU message thread
        imuIntegratorOpt_ = new gtsam::PreintegratedImuMeasurements(p, prevBias_); // setting up the IMU integration for optimization
        // imuIntegratorImu_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias);

        lastImuTime = -1;

        key = 1;

    }

    // bool initMeanVelocity()
    // {
    //     if (imu_queue.size() < 100)
    //         return false;

    //     for (int i = 0; i < 100; i++)
    //     {
    //         // 1 second imu data used for mean velocity initialization
    //         imuMeanAcc += (imu_queue[i].linear_acc / 100.0);
    //         imu_queue.pop_front();
    //     }


    //     boost::shared_ptr<gtsam::PreintegrationParams> p = gtsam::PreintegrationParams::MakeSharedU(0.0);

    //     p->accelerometerCovariance = gtsam::Matrix33::Identity(3, 3) * pow(imuAccNoise, 2);             // acc white noise in continuous
    //     p->gyroscopeCovariance = gtsam::Matrix33::Identity(3, 3) * pow(imuGyrNoise, 2);                 // gyro white noise in continuous
    //     p->integrationCovariance = gtsam::Matrix33::Identity(3, 3) * pow(1e-4, 2);                      // error committed in integrating position from velocities
    //     gtsam::imuBias::ConstantBias prior_imu_bias((gtsam::Vector(6) << 0, 0, 0, 0, 0, 0).finished()); // assume zero initial bias

    //     prevBias_ = gtsam::imuBias::ConstantBias((gtsam::Vector(6) << 0, 0, 0, 0, 0, 0).finished());
    //     priorPoseNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2).finished()); // rad,rad,rad,m, m, m
    //     priorVelNoise = gtsam::noiseModel::Isotropic::Sigma(3, 1e2);                                                               // m/s
    //     priorBiasNoise = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3);                                                             // 1e-2 ~ 1e-3 seems to be good
    //     correctionNoise = gtsam::noiseModel::Isotropic::Sigma(6, 1e-2);                                                            // meter
    //     noiseModelBetweenBias = (gtsam::Vector(6) << imuAccBiasN, imuAccBiasN, imuAccBiasN, imuGyrBiasN, imuGyrBiasN, imuGyrBiasN).finished();

    //     imuIntegratorImu_ = new gtsam::PreintegratedImuMeasurements(p, prevBias_); // setting up the IMU integration for IMU message thread
    //     imuIntegratorOpt_ = new gtsam::PreintegratedImuMeasurements(p, prevBias_); // setting up the IMU integration for optimization
    //     // imuIntegratorImu_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias);

    //     lastImuTime = -1;

    //     key = 1;

    //     std::cout << "IMU Initialized!" << std::endl;

    //     return true;
    // }

    void resetOptimization()
    {
        gtsam::ISAM2Params optParameters;
        optParameters.relinearizeThreshold = 0.1;
        optParameters.relinearizeSkip = 1;
        optimizer = gtsam::ISAM2(optParameters);

        gtsam::NonlinearFactorGraph newGraphFactors;
        graphFactors = newGraphFactors;

        gtsam::Values NewGraphValues;
        graphValues = NewGraphValues;
    }

    void resetParams()
    {
        lastImuTime = -1;
        // doneFirstOpt = false;
        systemInitialized = false;
    }

    void integrateIMU(const ImuMessage &imu_msg, Eigen::Matrix4d &Tiw, Eigen::Vector3d &imuVelocity);
    void integrateIMU(const std::vector<ImuMessage> &imu_msg_vec, Eigen::Matrix4d &T_imu_rel, Eigen::Vector3d &imuVelocity);
    void optimizeBias(const Eigen::Matrix4d T_wl, const Eigen::Matrix4d T_il);
    bool failureDetection(const gtsam::Vector3& velCur, const gtsam::imuBias::ConstantBias& biasCur);
    void resetIntegrator();
    


    // void initializeIMU(const std::vector<std::vector<double>> imuMeasurementVec);
};

#endif