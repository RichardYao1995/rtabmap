#include "rtabmap/core/DBDriver.h"
#include "rtabmap/core/Rtabmap.h"
#include "rtabmap/core/util3d.h"
#include "rtabmap/core/util3d_filtering.h"
#include "rtabmap/core/util3d_transforms.h"
#include "rtabmap/core/util3d_surface.h"
#include "rtabmap/utilite/UMath.h"
#include "rtabmap/utilite/UTimer.h"
#include "rtabmap/utilite/UFile.h"
#include "pcl/filters/filter.h"
#include "pcl/io/ply_io.h"
#include "pcl/io/obj_io.h"
#include "pcl/common/common.h"
#include "pcl/surface/poisson.h"
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/opencv.hpp"
#include <stdio.h>
#include <chrono>
#include <fstream>

using namespace rtabmap;
using namespace cv;
using namespace std;

void pose_estimation_3d3d (const std::vector<cv::Point3f> &pts1, const std::vector<cv::Point3f> &pts2)
{
    cv::Mat R;
    cv::Mat t;
    cv::Point3f p1, p2;     // center of mass
    int N = pts1.size();
    for(int i = 0;i < N;i++)
    {
        p1 += pts1[i];
        p2 += pts2[i];
    }
    p1 = cv::Point3f(cv::Vec3f(p1) / N);
    p2 = cv::Point3f(cv::Vec3f(p2) / N);
    std::vector<cv::Point3f> q1(N), q2(N); // remove the center
    for(int i = 0;i < N;i++)
    {
        q1[i] = pts1[i] - p1;
        q2[i] = pts2[i] - p2;
    }

    // compute q1*q2^T
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for(int i = 0;i < N;i++)
    {
        W += Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) * Eigen::Vector3d(q2[i].x, q2[i].y, q2[i].z).transpose();
    }

    // SVD on W
    Eigen::JacobiSVD<Eigen::Matrix3d> svd (W, Eigen::ComputeFullU|Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    if(U.determinant() * V.determinant() < 0)
    {
        for(int x = 0;x < 3;++x)
        {
            U(x, 2) *= -1;
        }
    }

    Eigen::Matrix3d R_ = U * (V.transpose());
    Eigen::Vector3d t_ = Eigen::Vector3d(p1.x, p1.y, p1.z) - R_ * Eigen::Vector3d(p2.x, p2.y, p2.z);

    // convert to cv::Mat
    R = ( Mat_<double> ( 3,3 ) <<
          R_ ( 0,0 ), R_ ( 0,1 ), R_ ( 0,2 ),
          R_ ( 1,0 ), R_ ( 1,1 ), R_ ( 1,2 ),
          R_ ( 2,0 ), R_ ( 2,1 ), R_ ( 2,2 )
        );
    t = ( Mat_<double> ( 3,1 ) << t_ ( 0,0 ), t_ ( 1,0 ), t_ ( 2,0 ) );
    cout << "R=" << R_<<endl;
    cout <<"t:"<<t_<<endl;
    for ( int i=0; i<N; i++ )
    {
        cout<<R_*Eigen::Vector3d(pts2[i].x,pts2[i].y,pts2[i].z) +t_<<endl ;
    }
    Eigen::Matrix4f T;
    T<<       R_(0,0),R_(0,0),R_(0,0),t_(0,0),
         R_(0,0),R_(0,0),R_(0,0),t_(0,0),
         R_(0,0),R_(0,0),R_(0,0),t_(0,0),
         R_(0,0),R_(0,0),R_(0,0),t_(0,0);
    std::cout << T << endl;
}

int main(int argc, char * argv[])
{
//    vector<Point3f> pts1,pts2,pts3;
//    pts1.push_back(Point3f(4.96169,91.0373,3.01159));
//    pts1.push_back(Point3f(-0.257996,104.099,3.22297));
//    pts1.push_back(Point3f(-1.67362,122.565,10.2881));
//    pts1.push_back(Point3f(-21.119,135.445,9.09069));
//    pts1.push_back(Point3f(7.73983,94.172,6.55096));
////    pts1.push_back(Point3f(-79.1187,88.9439,0.216082));
////    pts1.push_back(Point3f(-65.0361,55.9337,-0.106155));
////    pts1.push_back(Point3f(-88.5016,76.9515,0.136547));
////    pts1.push_back(Point3f(-75.0379,80.4143,0.152494));

//    pts2.push_back(Point3f(4.68259,92.402,2.99382));
//    pts2.push_back(Point3f(0.0416412,104.626,3.2056));
//    pts2.push_back(Point3f(-2.23943,123.344,10.1006));
//    pts2.push_back(Point3f(-20.7243,135.601,9.18349));
//    pts2.push_back(Point3f(7.74815,95.1643,6.48832));
//    pts2.push_back(Point3f(-78.9388,88.7533,0.57402));
//    pts2.push_back(Point3f(-66.9274,58.8845,0.876542));
//    pts2.push_back(Point3f(-88.6643,77.4449,0.555642));
//    pts2.push_back(Point3f(-73.9631,79.203,0.601167));

//    pts3.push_back(Point3f(-54.5047,52.127,19.5487));
//    pts3.push_back(Point3f(-89.1415,83.9021,0.183912));
//    pts3.push_back(Point3f(-77.787,82.6231,0.368408));
//    pts3.push_back(Point3f(-94.1568,93.6053,0.918437));
//    pts3.push_back(Point3f(-66.8343,63.2878,7.08069));
//    pts3.push_back(Point3f(-79.0409,88.9732,0.212239));
//    pts3.push_back(Point3f(-67.0137,59.1095,0.0679692));
//    pts3.push_back(Point3f(-88.761,77.6602,0.204892));
//    pts3.push_back(Point3f(-74.061,79.4259,0.0822068));


//    pose_estimation_3d3d(pts1, pts2);
//      (4.6144,91.5962,3.06103)
//      (0.0523562,103.85,3.2532)
//      (-2.14724,122.578,10.1468)
//      (-20.5457,134.957,9.12771)
//      (7.6771,94.3362,6.57553)




    ULogger::setType(ULogger::kTypeConsole);
    ULogger::setLevel(ULogger::kError);

    bool isOptimate = 0;
    bool isGlobal   = 1;
    int maxDistance = 20.f;

    std::cout<< "isOptimate:" <<isOptimate<<std::endl;
    std::cout<< "isGlobal:"   <<isGlobal<<std::endl;
    std::cout<< "maxDistance:"<<maxDistance<<std::endl;

    std::string dbPath = argv[argc-1];
    auto db = dbPath.rfind(".db");
    std::string date = dbPath.substr(db - 13, 6);

    // Get parameters
    ParametersMap parameters;
    DBDriver * driver = DBDriver::create();
    if(driver->openConnection(dbPath))
    {
        parameters = driver->getLastParameters();
        driver->closeConnection(false);
    }
    else
    {
        UERROR("Cannot open database %s!", dbPath.c_str());
    }
    delete driver;

    // Get the global optimized map
    Rtabmap rtabmap;
    rtabmap.init(parameters, dbPath);

    std::map<int, Signature> nodes;
    std::map<int, Transform> optimizedPoses;
    std::multimap<int, Link> links;
    rtabmap.get3DMap(nodes, optimizedPoses, links, isOptimate, isGlobal);

    std::map<int, Transform>::iterator iter=optimizedPoses.begin();
    for(size_t i = 0;i < optimizedPoses.size() / 4;i++)
        iter++;
    Signature node = nodes.find(iter->first)->second;
    node.sensorData().uncompressData();
    //std::vector<CameraModel> models = node.sensorData().cameraModels();

    cv::Mat image0 = node.sensorData().imageRaw();
    cv::imwrite("image/first_left.png", image0);
    cv::Mat image01 = node.sensorData().rightRaw();
    cv::imwrite("image/first_right.png", image01);
    std::cout << iter->second.rotationMatrix() << std::endl;

    pcl::IndicesPtr indices1(new std::vector<int>);
    std::vector<int> index1;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud = util3d::cloudRGBFromSensorData(
            node.sensorData(),
            4,           // image decimation before creating the clouds
            20,        // maximum depth of the cloud
            0.0f,
            indices1.get());
    pcl::removeNaNFromPointCloud(*cloud, *cloud, index1);
    pcl::PLYWriter writer1;
    cloud = util3d::transformPointCloud(cloud,iter->second);
    writer1.write("ply/first.ply", *cloud);

    iter++;
    node = nodes.find(iter->first)->second;
    node.sensorData().uncompressData();
    cv::Mat image1 = node.sensorData().imageRaw();
    cv::imwrite("image/second_left.png", image1);
    cv::Mat image02 = node.sensorData().rightRaw();
    cv::imwrite("image/second_right.png", image02);
    std::cout << iter->second.rotationMatrix() << std::endl;

    pcl::IndicesPtr indices2(new std::vector<int>);
    std::vector<int> index2;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud2 = util3d::cloudRGBFromSensorData(
            node.sensorData(),
            4,           // image decimation before creating the clouds
            20,        // maximum depth of the cloud
            0.0f,
            indices2.get());
    pcl::removeNaNFromPointCloud(*cloud2, *cloud2, index2);
    pcl::PLYWriter writer2;
    cloud2 = util3d::transformPointCloud(cloud2,iter->second);
    writer2.write("ply/second.ply", *cloud2);
    return 0;
}
