#pragma once
#ifndef IOUTILS_H
#define IOUTILS_H

#include <Common.h>
#include <pcl/common/transforms.h>
#include <pcl/point_types.h>
#include <dirent.h>

std::vector<std::string> getFileNames(const std::string& folder_path);

void readTextFile(std::string txt_path,
                  std::vector<std::string> &lines_vec);

void loadLaserDataset(std::string data_dir,
                      std::deque<LidarMessage> &lidar_queue);

void loadImageDataset(std::string data_dir,
                      std::deque<ImgMessage> &img_queue);

void loadImuDataset(std::string data_path,
                    std::deque<ImuMessage> &imu_queue);

void readLaserScan(std::string file_path,
                   pcl::PointCloud<PointType>::Ptr &cloud);

#endif