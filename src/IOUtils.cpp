#include <IOUtils.h>
#include <pcl/io/pcd_io.h>

std::vector<std::string> getFileNames(const std::string &folder_path)
{
    std::vector<std::string> file_names;
    DIR *dir = opendir(folder_path.c_str());
    if (dir)
    {
        dirent *entry;
        while ((entry = readdir(dir)) != nullptr)
        {
            if (entry->d_type == DT_REG) // regular file
            {
                file_names.push_back(entry->d_name);
            }
        }
        closedir(dir);
    }
    return file_names;
}

void readTextFile(std::string txt_path, std::vector<std::string> &lines_vec)
{
    std::ifstream file(txt_path);

    std::string line;

    while (getline(file, line))
        lines_vec.push_back(line);

    file.close();
}

void loadLaserDataset(std::string data_dir,
                      std::deque<LidarMessage> &lidar_queue)
{
    std::vector<std::string> file_list_unordered;
    std::vector<double> file_time_unordered;

    std::vector<std::string> file_names = getFileNames(data_dir);

    for (auto &entry : file_names)
        file_list_unordered.push_back(data_dir + "/" + entry);

    for (auto &file_dir : file_list_unordered)
    {
        std::vector<std::string> splited_strings = split(file_dir, '/');
        std::vector<std::string> splited_strings2 = split(splited_strings[splited_strings.size() - 1], '.');
        std::string time = splited_strings2[0] + "." + splited_strings2[1];
        file_time_unordered.push_back(std::stod(time));
    }

    std::vector<size_t> order = argsort_d(file_time_unordered);

    for (size_t i = 0; i < order.size(); i++)
    {
        LidarMessage lidar_msg(file_list_unordered[order[i]], file_time_unordered[order[i]]);
        lidar_queue.push_back(lidar_msg);
    }
}

void loadImageDataset(std::string data_dir,
                      std::deque<ImgMessage> &img_queue)
{
    std::vector<std::string> file_list_unordered;
    std::vector<double> file_time_unordered;

    std::vector<std::string> file_names = getFileNames(data_dir);

    for (auto &entry : file_names)
        file_list_unordered.push_back(data_dir + "/" + entry);

    for (auto &file_dir : file_list_unordered)
    {
        std::vector<std::string> splited_strings = split(file_dir, '/');
        std::vector<std::string> splited_strings2 = split(splited_strings[splited_strings.size() - 1], '.');
        std::string time = splited_strings2[0] + "." + splited_strings2[1];
        file_time_unordered.push_back(std::stod(time));
    }

    std::vector<size_t> order = argsort_d(file_time_unordered);

    for (size_t i = 0; i < order.size(); i++)
    {
        ImgMessage img_msg(file_list_unordered[order[i]], file_time_unordered[order[i]]);
        img_queue.push_back(img_msg);
    }
}

void loadImuDataset(std::string data_path, std::deque<ImuMessage> &imu_queue)
{
    std::vector<std::string> lines_vec;
    readTextFile(data_path, lines_vec);

    for (size_t i = 0; i < lines_vec.size(); i++)
    {
        std::vector<std::string> data = split(lines_vec[i], ' ');

        std::vector<double> line;

        for (int j = 0; j < 7; j++)
            line.push_back(std::stod(data[j]));

        ImuMessage msg(Eigen::Vector3d(line[1], line[2], line[3]), Eigen::Vector3d(line[4], line[5], line[6]), line[0]);
        imu_queue.push_back(msg);
    }
}

void readLaserScan(std::string file_path,
                   pcl::PointCloud<PointType>::Ptr &cloud)

{

    if (pcl::io::loadPCDFile<PointType> (file_path, *cloud) == -1) // load the file
    {
        PCL_ERROR ("Couldn't read file \n");
        return;
    }

    // std::string line;
    // std::ifstream file(file_path);
    // if (file.is_open())
    // {
    //     while (getline(file, line))
    //     {
    //         std::vector<std::string> data = split(line, ' ');
    //         Eigen::Vector3d xyz(std::stof(data[0]), std::stof(data[1]), std::stof(data[2]));
    //         double intensity = std::stof(data[4]);
    //         PointType p;
    //         p.x = xyz[0];
    //         p.y = xyz[1];
    //         p.z = xyz[2];
    //         p.intensity = intensity;

    //         // double offset_time = std::stod(data[3]); // 0~1
    //         // p.offset_time = offset_time / (100000000.0);

    //         cloud->push_back(p);
    //     }
    //     file.close();
    // }
    // else
    // {
    //     std::cout << "Unable to open file";
    //     return;
    // }
}