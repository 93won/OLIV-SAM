#pragma once

#include <mutex>
#include <thread>
#include <deque>
#include <unordered_map>

// Graphs
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/ISAM2.h>

// Factors
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/slam/ProjectionFactor.h>
#include <gtsam/slam/SmartProjectionPoseFactor.h>

#include <Tracker.h>
#include <DataStructure.h>

using gtsam::symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)
using gtsam::symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using gtsam::symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)

typedef gtsam::SmartProjectionPoseFactor<gtsam::Cal3_S2> SmartFactor;

#include <gtsam/geometry/Pose3.h>
#include <gtsam/navigation/ImuBias.h>

class GraphSolver
{
public:
    GraphSolver()
    {

        graph = new gtsam::NonlinearFactorGraph();
        graphNew = new gtsam::NonlinearFactorGraph();

        // ISAM2 solver
        gtsam::ISAM2Params isamParams;
        isamParams.relinearizeThreshold = 0.01;
        isamParams.relinearizeSkip = 1;
        isamParams.cacheLinearizedFactors = false;
        isamParams.enableDetailedResults = true;
        isamParams.print();
        isam2 = new gtsam::ISAM2(isamParams);
    }

    void initialize(double timestamp);
    gtsam::CombinedImuFactor createImuFactor(const std::vector<ImuMessage> &imuMsgVec);
    void optimizeTracker(const Tracker::Ptr &tracker);

    void resetImuPreintegration();

    Eigen::Vector3d imuMeanAcc;
    gtsam::NonlinearFactorGraph *graphNew;
    gtsam::NonlinearFactorGraph *graph;
    gtsam::Values valuesNew;
    gtsam::Values valuesInitial;
    gtsam::ISAM2 *isam2;


    bool systeminitalized = false;
    gtsam::PreintegratedCombinedMeasurements *preintGTSAM;

    int key = 0;
    double sigma_prior_rotation = 0.1;
    double sigma_prior_translation = 0.3;
    double sigma_velocity = 0.1;
    double sigma_bias = 0.15;
    double sigma_pose_rotation = 0.1;
    double sigma_pose_translation = 0.2;
    double accelerometer_noise_density = 0.1;
    double gyroscope_noise_density = 0.01;
    double accelerometer_random_walk = 0.01;
    double gyroscope_random_walk = 0.001;






};
