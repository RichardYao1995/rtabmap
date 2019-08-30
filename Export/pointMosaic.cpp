#include <iostream>

#include <iostream>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/locale.hpp>
#include <boost/program_options.hpp>
#include <boost/progress.hpp>
#include <boost/optional/optional.hpp>
#include <vector>
#include <boost/tokenizer.hpp>
#include <eigen3/Eigen/Dense>
#include <stdlib.h>
#include <iomanip>

#include <pcl/common/transforms.h>
#include<pcl/point_cloud.h>
//#include<pcl//pcl_conversions.h>
#include<pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>


#include <rtabmap/core/DBDriver.h>
#include <rtabmap/core/Rtabmap.h>
#include <rtabmap/core/util3d.h>
#include <rtabmap/core/util3d_filtering.h>
#include <rtabmap/core/util3d_transforms.h>
#include <rtabmap/core/util3d_surface.h>
#include <rtabmap/utilite/UMath.h>
#include <rtabmap/utilite/UTimer.h>
#include <rtabmap/utilite/UFile.h>

typedef Eigen::Matrix<double,1,6> POSE;
typedef pcl::PointXYZRGB PointType;
//typedef pcl::PointXYZI PointType;
void read_ref_pose(const std::string& ref_path, std::map<int,POSE>& ref_pose)
{
    boost::optional<POSE> start_pose;
    std::ifstream fin_ref_pose(ref_path.c_str());

    std::string line;
    int o = 0;
    while (getline(fin_ref_pose, line))
    {
        std::vector<double> data_vec;
        // split(line,std::string(","),&data_vec);

        int n = static_cast<int>(line.size());
        for (int i = 0; i < n; ++i){
            if (line[i] == ','){
                line[i] = ' ';
            }
        }
        std::stringstream record(line);
        for(int i = 0; i < 8 ; ++i)
        {
            double tem;
            record >> tem;
            data_vec.push_back(tem);
        }
        if(o == 0)
        {
            std::cout<<data_vec[0]<<",";
            std::cout<<data_vec[1]<<",";
            std::cout<<data_vec[2]<<",";
            std::cout<<data_vec[3]<<",";
            std::cout<<data_vec[4]<<",";
            std::cout<<data_vec[5]<<",";
            std::cout<<data_vec[6]<<",";
            std::cout<<data_vec[7]<<","
            <<std::endl;
            o = 1;
        }

        POSE ref_xyz_rpy;
        ref_xyz_rpy(0) = data_vec[1];
        ref_xyz_rpy(1) = data_vec[2];
        ref_xyz_rpy(2) = data_vec[3];
        ref_xyz_rpy(3) = data_vec[4];
        ref_xyz_rpy(4) = data_vec[5];
        ref_xyz_rpy(5) = data_vec[6];


        assert(data_vec.size() == 8);

        if (!start_pose)
        {
            start_pose = ref_xyz_rpy;
        }

        ref_xyz_rpy(0) -= (*start_pose)(0);
        ref_xyz_rpy(1) -= (*start_pose)(1);
        ref_xyz_rpy(2) -= (*start_pose)(2);

        ref_pose.insert({data_vec[0],ref_xyz_rpy});
    }
    fin_ref_pose.close();
}

void writePcd(const std::string& output,pcl::PointCloud<PointType>::Ptr& cloud)
{
    pcl::PCDWriter writer;

    pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames;
    pcl::PointCloud<PointType>::Ptr globalLidarDS (new pcl::PointCloud<PointType>);
    downSizeFilterGlobalMapKeyFrames.setLeafSize(0.1f, 0.1f, 0.1f);
    downSizeFilterGlobalMapKeyFrames.setInputCloud(cloud);
    downSizeFilterGlobalMapKeyFrames.filter(*globalLidarDS);

    writer.write(output,*globalLidarDS);
}
Eigen::Matrix3d getRFromrpy(const Eigen::Vector3d& rpy)
{
    Eigen::Matrix3d R;
    Eigen::Vector3d ea(rpy(0),rpy(1),rpy(2));
    R = Eigen::AngleAxisd(ea[2], Eigen::Vector3d::UnitZ()) *
                 Eigen::AngleAxisd(ea[1], Eigen::Vector3d::UnitY()) *
                 Eigen::AngleAxisd(ea[0], Eigen::Vector3d::UnitX());
    return R;
}
Eigen::Matrix3d getRFromrpy_kitti(const Eigen::Vector3d& rpy)
{
    Eigen::Matrix3d R;
    Eigen::Vector3d ea(rpy(0),rpy(1),rpy(2));
//    R = Eigen::AngleAxisd(ea[0], Eigen::Vector3d::UnitX()) *
//                 Eigen::AngleAxisd(ea[1], Eigen::Vector3d::UnitY()) *
//                 Eigen::AngleAxisd(ea[2], Eigen::Vector3d::UnitZ());
    double rx = rpy(0);
    double ry = rpy(1);
    double rz = rpy(2);
    Eigen::Matrix3d Rx,Ry,Rz;
    Rx << 1, 0 , 0 ,
          0,cos(rx),-sin(rx),
          0 ,sin(rx),cos(rx);
    Ry << cos(ry), 0 , sin(ry) ,
          0 ,      1 ,    0 ,
          -sin(ry),0,cos(ry);
    Rz << cos(rz) , -sin(rz) , 0,
          sin(rz),  cos(rz) ,  0,
          0,              0,   1;
//    rx = oxts{i}(4); % roll
//    ry = oxts{i}(5); % pitch
//    rz = oxts{i}(6); % heading

//    Rx = [1 0 0; 0 cos(rx) -sin(rx); 0 sin(rx) cos(rx)]; % base => nav (level oxts => rotated oxts)
//    Ry = [cos(ry) 0 sin(ry); 0 1 0; -sin(ry) 0 cos(ry)]; % base => nav (level oxts => rotated oxts)
//    Rz = [cos(rz) -sin(rz) 0; sin(rz) cos(rz) 0; 0 0 1]; % base => nav (level oxts => rotated oxts)
//    R = Rz*Ry*Rx;
    return Rz*Ry*Rx;
}



pcl::PointCloud<PointType>::Ptr transformPointCloud(
        const pcl::PointCloud<PointType>::Ptr & cloud,
        const Eigen::Matrix4f & transform)
{
    pcl::PointCloud<PointType>::Ptr output(new pcl::PointCloud<PointType>);
    pcl::transformPointCloud(*cloud, *output, transform);
    return output;
}

pcl::PointCloud<PointType>::Ptr transformPointCloud(
        const pcl::PointCloud<PointType>::Ptr & cloud,
        const Eigen::Matrix3d& R,
        const Eigen::Vector3d& t)
{
    Eigen::Matrix4d M;
    M(0,0) = R(0,0);
    M(0,1) = R(0,1);
    M(0,2) = R(0,2);
    M(1,0) = R(1,0);
    M(1,1) = R(1,1);
    M(1,2) = R(1,2);
    M(2,0) = R(2,0);
    M(2,1) = R(2,1);
    M(2,2) = R(2,2);

    M(0,3) = t(0);
    M(1,3) = t(1);
    M(2,3) = t(2);

    M(3,0) = 0;
    M(3,1) = 0;
    M(3,2) = 0;
    M(3,3) = 1;
    pcl::PointCloud<PointType>::Ptr output(new pcl::PointCloud<PointType>);
    pcl::transformPointCloud(*cloud, *output, M.cast<float>());
    return output;
}
void loadPoints(const std::string name, const std::string depthDir, const std::string imageDir,pcl::PointCloud<PointType>::Ptr& cloud)
{
    std::string rgbName = imageDir+name+".tiff";
    std::string depthName = depthDir+name+".png";
    cv::Mat rgb = cv::imread(rgbName,cv::IMREAD_ANYDEPTH);
    cv::Mat depth = cv::imread(depthName,cv::IMREAD_ANYDEPTH);
    cloud = rtabmap::util3d::cloudFromDepthRGB(rgb,depth,640.f,360.f,762.82f,762.82f,4,20);
}

int main()
{
    std::string ref_path = "/home/uisee/Documents/dockerfiles/ref_pose.txt";
    std::string pointDir = "/home/uisee/Documents/dataBase/simulate_618/xyzrgb/";
    std::string rgbDir = "/home/uisee/Documents/dataBase/simulate_618/left/";
    std::string depthDir = "/home/uisee/Documents/dataBase/simulate_618/depth/";
    std::string output_path = "result.pcd";

    std::map<int,POSE> ref_pose;
    read_ref_pose(ref_path,ref_pose);

    pcl::PointCloud<PointType>::Ptr cloud_all(new pcl::PointCloud<PointType>);

    int count(80);
    for(auto it:ref_pose)
    {
        int index = it.first;
        if(index < 80)
            continue;
        std::string indexName = std::to_string(index);

        while (indexName.size()<10)
        {
            indexName="0"+indexName;
        }
        std::string pointPath = pointDir+"img_map"+indexName+"..pcd";
        pcl::PointCloud<PointType>::Ptr cloud_in_camera(new pcl::PointCloud<PointType>);
        pcl::PointCloud<PointType>::Ptr cloud_in_car(new pcl::PointCloud<PointType>);

        loadPoints(indexName,depthDir,rgbDir,cloud_in_camera);
//        pcl::io::loadPCDFile (pointPath, *cloud_in_camera);

        std::cout<<cloud_in_camera->size()<<std::endl;
        Eigen::Matrix4d T_image_to_car;
        T_image_to_car << 1 , 0 , 0 , -0.175,
                          0 , 0 , 1 , 1.99 ,
                          0 , -1, 0 , 0,
                          0 , 0 , 0 , 1;

        POSE& pose = it.second;
        Eigen::Vector3d X (pose(0),pose(1),pose(2));
        Eigen::Matrix3d R = getRFromrpy(Eigen::Vector3d(pose(3),pose(4),pose(5)));

        *cloud_in_car = *transformPointCloud(cloud_in_camera, T_image_to_car.cast<float>());
        *cloud_all   += *transformPointCloud(cloud_in_car, R,X);


        if(count > 500)
            break;
        count++;

    }

    writePcd(output_path,cloud_all);

    return 0;

}

void read_ref_pose_kiiti(const std::string& ref_path, std::map<double,POSE>& ref_pose)
{
    boost::optional<POSE> start_pose;
    std::ifstream fin_ref_pose(ref_path.c_str());

    std::string line;
    int o = 0;
    while (getline(fin_ref_pose, line))
    {
        std::vector<double> data_vec;
        // split(line,std::string(","),&data_vec);

        int n = static_cast<int>(line.size());
        for (int i = 0; i < n; ++i){
            if (line[i] == ','){
                line[i] = ' ';
            }
        }
        std::stringstream record(line);
        for(int i = 0; i < 7 ; ++i)
        {
            double tem;
            record >> tem;
            data_vec.push_back(tem);
        }
        if(o == 0)
        {
            std::cout<<data_vec[0]<<",";
            std::cout<<data_vec[1]<<",";
            std::cout<<data_vec[2]<<",";
            std::cout<<data_vec[3]<<",";
            std::cout<<data_vec[4]<<",";
            std::cout<<data_vec[5]<<",";
            std::cout<<data_vec[6]<<","
            <<std::endl;
            o = 1;
        }

        POSE ref_xyz_rpy;
        ref_xyz_rpy(0) = data_vec[1];
        ref_xyz_rpy(1) = data_vec[2];
        ref_xyz_rpy(2) = data_vec[3];
        ref_xyz_rpy(3) = data_vec[4];
        ref_xyz_rpy(4) = data_vec[5];
        ref_xyz_rpy(5) = data_vec[6];


        assert(data_vec.size() == 7);

        if (!start_pose)
        {
            start_pose = ref_xyz_rpy;
        }

        ref_xyz_rpy(0) -= (*start_pose)(0);
        ref_xyz_rpy(1) -= (*start_pose)(1);
        ref_xyz_rpy(2) -= (*start_pose)(2);

        ref_pose.insert({data_vec[0],ref_xyz_rpy});
    }
    fin_ref_pose.close();
}
void read_ref_pose_kiiti_xyz(const std::string& ref_path, std::map<double,POSE>& ref_pose)
{
    boost::optional<POSE> start_pose;
    std::ifstream fin_ref_pose(ref_path.c_str());

    std::string line;
    int o = 0;
    while (getline(fin_ref_pose, line))
    {
        std::vector<double> data_vec;
        // split(line,std::string(","),&data_vec);

        int n = static_cast<int>(line.size());
        for (int i = 0; i < n; ++i){
            if (line[i] == ','){
                line[i] = ' ';
            }
        }
        std::stringstream record(line);
        for(int i = 0; i < 4 ; ++i)
        {
            double tem;
            record >> tem;
            data_vec.push_back(tem);
        }
        if(o == 0)
        {
            std::cout<<data_vec[0]<<",";
            std::cout<<data_vec[1]<<",";
            std::cout<<data_vec[2]<<",";
            std::cout<<data_vec[3]<<","
            <<std::endl;
            o = 1;
        }

        POSE ref_xyz_rpy;
        ref_xyz_rpy(0) = data_vec[1];
        ref_xyz_rpy(1) = data_vec[2];
        ref_xyz_rpy(2) = data_vec[3];


        assert(data_vec.size() == 4);

        if (!start_pose)
        {
            start_pose = ref_xyz_rpy;
        }

        ref_xyz_rpy(0) -= (*start_pose)(0);
        ref_xyz_rpy(1) -= (*start_pose)(1);
        ref_xyz_rpy(2) -= (*start_pose)(2);

        ref_pose.insert({data_vec[0],ref_xyz_rpy});
//        std::cout<<std::setprecision(20)<<(int)data_vec[0]<<std::endl;
    }
    fin_ref_pose.close();
}
void read_ref_pose_kiiti_rpy(const std::string& ref_path, std::map<double,POSE>& ref_pose)
{
    boost::optional<POSE> start_pose;
    std::ifstream fin_ref_pose(ref_path.c_str());

    std::string line;
    int o = 0;
    while (getline(fin_ref_pose, line))
    {
        std::vector<double> data_vec;
        // split(line,std::string(","),&data_vec);

        std::stringstream record(line);
        for(int i = 0; i < 3 ; ++i)
        {
            double tem;
            record >> tem;
            data_vec.push_back(tem);
        }
        if(o == 0)
        {
            std::cout<<data_vec[0]<<",";
            std::cout<<data_vec[1]<<",";
            std::cout<<data_vec[2]<<","
            <<std::endl;
        }
        auto it = ref_pose.begin();
        std::advance(it,o);
        it->second(3) = data_vec[0];
        it->second(4) = data_vec[1];
        it->second(5) = data_vec[2];

        o++;
    }
    fin_ref_pose.close();
}
//void readBin(const std::string& file, pcl::PointCloud<PointType>& cloud)
//{
//    pcl::PointCloud<PointType>::Ptr points (new pcl::PointCloud<PointType>);
////    pcl::PointCloud<PointType> cloud;
//    using namespace std;
//    PointType point;
//    fstream input(file.c_str(), ios::in | ios::binary);
//    if(!input.good()){
//        std::cout << "Could not read file: " << file << endl;
//        exit(EXIT_FAILURE);
//    }
//    input.seekg(0, ios::beg);

//    for (int i = 0; input.good() && !input.eof(); i++) {
//        input.read((char *) &point.x, 3*sizeof(float));
//        input.read((char *) &point.intensity, sizeof(float));
//        cloud.push_back(point);
//    }
//    input.close();
//}

//int main()
//{
//    std::string ref_path_xyz = "/home/uisee/Documents/dataBase/2011_10_03/2011_10_03_drive_0027_sync/velodyne_points/kitti_utm_cord.txt";
//    std::string ref_path_rpy = "/home/uisee/Documents/dataBase/2011_10_03/2011_10_03_drive_0027_sync/velodyne_points/kitti_rpy.txt";
//    std::string pointDir = "/home/uisee/Documents/dataBase/2011_10_03/2011_10_03_drive_0027_sync/velodyne_points/data/";
//    std::string output_path = "result.pcd";

//    std::map<double,POSE> ref_pose;
//    read_ref_pose_kiiti_xyz(ref_path_xyz,ref_pose);
//    read_ref_pose_kiiti_rpy(ref_path_rpy,ref_pose);

//    pcl::PointCloud<PointType>::Ptr cloud_all(new pcl::PointCloud<PointType>);
//    std::cout<<ref_pose.size()<<std::endl;
//    int count(0);
//    for(auto it:ref_pose)
//    {
//        int index = count;
////        if(index < 80)
////            continue;
//        if(count > 1)
//            break;
//        count++;

//        std::string indexName = std::to_string(index);

//        while (indexName.size()<10)
//        {
//            indexName="0"+indexName;
//        }
//        std::string pointPath = pointDir+indexName+".bin";
//        pcl::PointCloud<PointType>::Ptr cloud_in_lidar(new pcl::PointCloud<PointType>);
//        pcl::PointCloud<PointType>::Ptr cloud_in_car(new pcl::PointCloud<PointType>);
//        std::cout<<pointPath<<std::endl;
//        readBin(pointPath,*cloud_in_lidar);

//        std::cout<<cloud_in_lidar->size()<<std::endl;
////        cloud_in_lidar->clear();
//        pcl::PointXYZI point(0);
//        cloud_in_lidar->push_back(point);
//        Eigen::Matrix4d T_imu_to_lidar;

//        T_imu_to_lidar << 9.999976e-01, 7.553071e-04, -2.035826e-03,8.086759e-01,
//                          -7.854027e-04, 9.998898e-01, -1.482298e-02, -3.195559e-01,
//                          2.024406e-03, 1.482454e-02, 9.998881e-01,   7.997231e-01,
//                          0  ,0  ,0  ,1;
//        Eigen::Matrix3d R_lidar_to_car,R_car_to_lidar;
//        R_car_to_lidar <<9.999976e-01, 7.553071e-04, -2.035826e-03,
//                         -7.854027e-04, 9.998898e-01, -1.482298e-02,
//                         2.024406e-03, 1.482454e-02, 9.998881e-01;

//        R_lidar_to_car = R_car_to_lidar.inverse();
//        Eigen::Vector3d X_car_to_lidar(-8.086759e-01,-3.195559e-01,7.997231e-01);


//        POSE& pose = it.second;
//        Eigen::Vector3d X (pose(0),pose(1),pose(2));
//        Eigen::Matrix3d R = getRFromrpy_kitti(Eigen::Vector3d(pose(3),pose(4),pose(5)));

//        std::cout<<X.transpose()<<std::endl;
//        std::cout<<Eigen::Vector3d(pose(3),pose(4),pose(5)).transpose()<<std::endl;

//        *cloud_in_car = *transformPointCloud(cloud_in_lidar, Eigen::Matrix3d::Identity(),Eigen::Vector3d::Zero());
//        *cloud_all   += *transformPointCloud(cloud_in_car, R,X);



//    }

//    writePcd(output_path,cloud_all);
//    std::string output_path_ply = "result.ply";
//    pcl::PLYWriter writerPly;

//    pcl::PCLPointCloud2 cloudIn;
//    if (pcl::io::loadPCDFile(output_path , cloudIn) < 0)
//    {
//        std::cout << "Error: cannot load the PCD file!!!"<< std::endl;
//        return -1;
//    }
//    writerPly.writeBinary(output_path_ply,cloudIn);

//    return 0;
//}


