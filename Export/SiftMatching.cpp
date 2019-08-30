
#include <iostream>
#include <vector>
#include "rtabmap/core/DBDriver.h"
#include "rtabmap/core/Rtabmap.h"
#include "rtabmap/core/util3d.h"
#include "rtabmap/core/util3d_filtering.h"
#include "rtabmap/core/util3d_transforms.h"
#include "rtabmap/core/util3d_surface.h"
#include "rtabmap/utilite/UMath.h"
#include "rtabmap/utilite/UTimer.h"
#include "rtabmap/utilite/UFile.h"
#include "rtabmap/core/util2d.h"
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/opencv.hpp"
#include "pcl/io/ply_io.h"
#include "pcl/io/obj_io.h"
#include "pcl/common/common.h"
#include "pcl/point_types.h"
#include "pcl/registration/icp.h"
#include "pcl/filters/filter.h"
#include "pcl/surface/poisson.h"
#include "Eigen/Core"
#include "Eigen/Dense"

using namespace cv;
using namespace std;

float focal = 762.72;

void distance_matching(const std::vector<cv::DMatch> &badMatch, std::vector<cv::DMatch> &goodMatch)
{
    double min_dist = badMatch[0].distance;
    for(unsigned int i = 1;i < badMatch.size();i++)
    {
        if(badMatch[i].distance < min_dist)
            min_dist = badMatch[i].distance;
    }

    //cout << "min_distance=" << min_dist << std::endl;

    for(unsigned int i = 0;i < badMatch.size();i++)
    {
        if(badMatch[i].distance < 3 * min_dist)
            goodMatch.push_back(badMatch[i]);
    }

    //std::cout << "size of bad match: " << badMatch.size() << std::endl
    //          << "size of good match: " << goodMatch.size() << std::endl;
}

void triangulation(const std::vector<cv::KeyPoint> &keypoint1,
                   const std::vector<cv::KeyPoint> &keypoint2,
                   const std::vector<cv::DMatch> &matches,
                   std::vector<cv::Point3d>& points)
{
    for(unsigned int i = 0;i < matches.size();i++)
    {
        cv::Point3d point;
        float x1 = keypoint1[matches[i].queryIdx].pt.x;
        float x2 = keypoint2[matches[i].trainIdx].pt.x;
        float y = keypoint1[matches[i].queryIdx].pt.y;
        if(x1 == x2)
            continue;
        point.x = 0.35 * (x1 - 640) / (x1 - x2);
        point.y = point.x * focal / (x1 - 640);
        if(point.y < 0 || point.y > 200)
            continue;
        point.z = point.y * (360 - y) / focal;
        points.push_back(point);
    }
}

Eigen::Matrix4f pose_estimation_3d3d (const pcl::PointCloud<pcl::PointXYZ>::Ptr pc1,
                                      const pcl::PointCloud<pcl::PointXYZ>::Ptr pc2)
{
    cv::Point3f p1, p2;     // center of mass
    int N = pc1->points.size();
    for(int i = 0;i < N;i++)
    {
        p1.x += pc1->points[i].x;
        p1.y += pc1->points[i].y;
        p1.z += pc1->points[i].z;
        p2.x += pc2->points[i].x;
        p2.y += pc2->points[i].y;
        p2.z += pc2->points[i].z;
    }
    p1 = cv::Point3f(cv::Vec3f(p1) / N);
    p2 = cv::Point3f(cv::Vec3f(p2) / N);
    std::vector<cv::Point3f> q1(N), q2(N); // remove the center
    for(int i = 0;i < N;i++)
    {
        q1[i].x = pc1->points[i].x - p1.x;
        q1[i].y = pc1->points[i].y - p1.y;
        q1[i].z = pc1->points[i].z - p1.z;
        q2[i].x = pc2->points[i].x - p2.x;
        q2[i].y = pc2->points[i].y - p2.y;
        q2[i].z = pc2->points[i].z - p2.z;
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

    Eigen::Matrix4f T;
    T << R_(0,0),R_(0,1),R_(0,2),t_(0,0),
         R_(1,0),R_(1,1),R_(1,2),t_(1,0),
         R_(2,0),R_(2,1),R_(2,2),t_(2,0),
         0, 0, 0, 1;
    return T;
}

Eigen::Matrix4f sift_matching(const cv::Mat &prev_image, const cv::Mat &prev_image_right,
                   const cv::Mat &next_image, const cv::Mat &next_image_right,
                   const rtabmap::Transform &trans1, const rtabmap::Transform &trans2)
//                   const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &scenePrev,
//                   const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &sceneNext)
{
        Eigen::Matrix4f transformation;
        transformation << 1, 0, 0, 0,
                      0, 1, 0, 0,
                      0, 0, 1, 0,
                      0, 0, 0, 1;

        cv::Mat imagePrev, imageNext, imagePrevRight, imageNextRight;
        cv::cvtColor(prev_image, imagePrev, CV_RGB2GRAY);
        cv::cvtColor(next_image, imageNext, CV_RGB2GRAY);
        imagePrevRight = prev_image_right;
        imageNextRight = next_image_right;

        //detect surf feature points
        cv::Ptr<cv::xfeatures2d::SIFT> siftDetector = cv::xfeatures2d::SIFT::create(400);
        std::vector<cv::KeyPoint> keyPointPrev, keyPointNext;
        std::vector<cv::KeyPoint> keyPointPrevRight, keyPointNextRight;
        siftDetector->detect(imagePrev, keyPointPrev);
        siftDetector->detect(imageNext, keyPointNext);
        siftDetector->detect(imagePrevRight, keyPointPrevRight);
        siftDetector->detect(imageNextRight, keyPointNextRight);

        //calculate surf descriptors
        cv::Mat imageDescPrev, imageDescNext;
        cv::Mat imageDescPrevRight, imageDescNextRight;
        siftDetector->compute(imagePrev, keyPointPrev, imageDescPrev);
        siftDetector->compute(imageNext, keyPointNext, imageDescNext);
        siftDetector->compute(imagePrevRight, keyPointPrevRight, imageDescPrevRight);
        siftDetector->compute(imageNextRight, keyPointNextRight, imageDescNextRight);

        //match next and previous left images and extract best 50 matches
        cv::FlannBasedMatcher matcherNextPrev;
        std::vector<cv::DMatch> matchPointsNextPrev;
        matcherNextPrev.match(imageDescPrev, imageDescNext, matchPointsNextPrev, cv::Mat());
        sort(matchPointsNextPrev.begin(), matchPointsNextPrev.end());
        matchPointsNextPrev.assign(matchPointsNextPrev.begin(), matchPointsNextPrev.begin() + 100);

        //find good matches in the best 50
        std::vector<cv::DMatch> matchPointsNextPrevGood;
        if(!matchPointsNextPrev.size())
            return transformation;
        distance_matching(matchPointsNextPrev, matchPointsNextPrevGood);

        //save top and good key points
        std::vector<cv::KeyPoint> topKeyPointPrev,topKeyPointNext;
        for (unsigned int i = 0; i < matchPointsNextPrevGood.size();i++)
        {
            topKeyPointPrev.push_back(keyPointPrev[matchPointsNextPrevGood[i].queryIdx]);
            topKeyPointNext.push_back(keyPointNext[matchPointsNextPrevGood[i].trainIdx]);
        }

        //compute top matched key points of previous and next left images, contain less than 50 key points
        cv::Mat imageDescPrevLeft, imageDescNextLeft;
        siftDetector->compute(imagePrev, topKeyPointPrev, imageDescPrevLeft);
        siftDetector->compute(imageNext, topKeyPointNext, imageDescNextLeft);

        //match previous left with top matched key points and right
        cv::BFMatcher matcherPrevLeftRight;
        std::vector<cv::DMatch> matchPointsPrevLeftRightBad;
        matcherPrevLeftRight.match(imageDescPrevLeft, imageDescPrevRight, matchPointsPrevLeftRightBad, cv::Mat());

        std::vector<cv::DMatch> matchPointsPrevLeftRightNotMatched;
        if(!matchPointsPrevLeftRightBad.size())
            return transformation;
        distance_matching(matchPointsPrevLeftRightBad, matchPointsPrevLeftRightNotMatched);

        //match next left with top matched key points and right
        cv::BFMatcher matcherNextLeftRight;
        std::vector<cv::DMatch> matchPointsNextLeftRightBad;
        matcherNextLeftRight.match(imageDescNextLeft, imageDescNextRight, matchPointsNextLeftRightBad, cv::Mat());

        std::vector<cv::DMatch> matchPointsNextLeftRightNotMatched;
        if(!matchPointsNextLeftRightBad.size())
            return transformation;
        distance_matching(matchPointsNextLeftRightBad, matchPointsNextLeftRightNotMatched);

        //compare two matches between previous and next
        unsigned int prev = 0, next = 0;
        std::vector<cv::DMatch> matchPointsNextLeftRight, matchPointsPrevLeftRight;
        while(prev < matchPointsPrevLeftRightNotMatched.size() && next < matchPointsNextLeftRightNotMatched.size())
        {
            if(matchPointsNextLeftRightNotMatched[next].queryIdx ==
               matchPointsPrevLeftRightNotMatched[prev].queryIdx)
            {
                matchPointsNextLeftRight.push_back(matchPointsNextLeftRightNotMatched[next]);
                matchPointsPrevLeftRight.push_back(matchPointsPrevLeftRightNotMatched[prev]);
                next++;
                prev++;
            }
            else
            {
                if(matchPointsNextLeftRightNotMatched[next].queryIdx <
                   matchPointsPrevLeftRightNotMatched[prev].queryIdx)
                    next++;
                else prev++;
            }
        }

//        cv::Mat matches;
//        cv::drawMatches(imagePrev, topKeyPointPrev, imagePrevRight, keyPointPrevRight, matchPointsPrevLeftRight,
//                        matches, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(),
//                        cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
//        cv::imwrite("result_prev.png", matches);
//        cv::drawMatches(imageNext, topKeyPointNext, imageNextRight, keyPointNextRight, matchPointsNextLeftRight,
//                        matches, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(),
//                        cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
//        cv::imwrite("result_next.png", matches);

        //calculate location of previous and next key points in camera coordinate
        std::vector<cv::Point3d> pointsPrev, pointsNext;
        triangulation(topKeyPointPrev, keyPointPrevRight, matchPointsPrevLeftRight, pointsPrev);
        triangulation(topKeyPointNext, keyPointNextRight, matchPointsNextLeftRight, pointsNext);

        //calculate point cloud of key points on previous and next images
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPrev(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloudNext(new pcl::PointCloud<pcl::PointXYZ>);
        for(unsigned int i = 0;i < pointsPrev.size();i++)
        {
            pcl::PointXYZ point;
            point.x = pointsPrev[i].x - 0.175;
            point.y = pointsPrev[i].y;
            point.z = pointsPrev[i].z + 1.99;

            cloudPrev->points.push_back(point);
            //std::cout << point << std::endl;
            point.x = pointsNext[i].x - 0.175;
            point.y = pointsNext[i].y;
            point.z = pointsNext[i].z + 1.99;
            cloudNext->points.push_back(point);
        }
        //std::cout << "number of good match: " << cloudPrev->points.size() << std::endl;
        cloudPrev = rtabmap::util3d::transformPointCloud(cloudPrev, trans1);
        cloudNext = rtabmap::util3d::transformPointCloud(cloudNext, trans2);

//        pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
//        icp.setMaxCorrespondenceDistance(2);
//        icp.setMaximumIterations(50);
//        icp.setTransformationEpsilon(1e-10);
//        icp.setEuclideanFitnessEpsilon(1e-8);
//        pcl::PointCloud<pcl::PointXYZ>::Ptr final(new pcl::PointCloud<pcl::PointXYZ>);

//        icp.setInputSource(cloudNext);
//        icp.setInputTarget(cloudPrev);
//        icp.align(*final);
//        std::cout << "score: " << icp.getFitnessScore() << std::endl
//                  << "is converged: " << icp.hasConverged() << std::endl;
//        for(unsigned int i = 0;i < cloudNext->points.size();i++)
//            std::cout << cloudPrev->points[i] << ' '
//                      << cloudNext->points[i] << std::endl;
//                      << final->points[i] << std::endl;
//        std::cout << icp.getFinalTransformation() << std::endl;
        if(cloudPrev->points.size() < 3)
        {
            return transformation;
        }

        transformation = pose_estimation_3d3d(cloudPrev, cloudNext);
        return transformation;
//        std::cout << "transformation: " << std::endl << transformation << std::endl;

//        pcl::PLYWriter writer1;
//        writer1.write("ply/previous.ply", *scenePrev);
//        writer1.write("ply/next.ply", *sceneNext);
//        pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed(new pcl::PointCloud<pcl::PointXYZRGB>);
//        pcl::transformPointCloud(*sceneNext, *transformed, transformation);
//        writer1.write("ply/transformed.ply", *transformed);
}

int main()
{
    rtabmap::ParametersMap parameters;
    rtabmap::DBDriver *driver = rtabmap::DBDriver::create();
    std::string dbPath("/home/uisee/Documents/RTAB-Map/190715-091852.db");
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
    rtabmap::Rtabmap rtabmap;
    rtabmap.init(parameters, dbPath);

    std::map<int, rtabmap::Signature> nodes;
    std::map<int, rtabmap::Transform> optimizedPoses;
    std::multimap<int, rtabmap::Link> links;
    rtabmap.get3DMap(nodes, optimizedPoses, links, 0, 1);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr Clouds(new pcl::PointCloud<pcl::PointXYZRGB>);

    std::map<int, rtabmap::Transform>::iterator iter=optimizedPoses.begin();
    rtabmap::Signature node = nodes.find(iter->first)->second;

    node.sensorData().uncompressData();
    rtabmap::Transform trans1 = iter->second;
    cv::Mat prev_image = node.sensorData().imageRaw();
    cv::Mat prev_image1 = node.sensorData().rightRaw();
//    pcl::IndicesPtr indices1(new std::vector<int>);
//    std::vector<int> index1;
//    pcl::PointCloud<pcl::PointXYZRGB>::Ptr scenePrev = rtabmap::util3d::cloudRGBFromSensorData(
//            node.sensorData(),
//            4,           // image decimation before creating the clouds
//            20,        // maximum depth of the cloud
//            0.0f,
//            indices1.get());
//    pcl::removeNaNFromPointCloud(*scenePrev, *scenePrev, index1);
//    scenePrev = rtabmap::util3d::transformPointCloud(scenePrev,iter->second);
    Eigen::Matrix4f transformation;
    Eigen::Matrix4f identity;
    identity<< 1, 0, 0, 0,
               0, 1, 0, 0,
               0, 0, 1, 0,
               0, 0, 0, 1;
    iter++;

    //Iteratively match each frame and add into point cloud
    for(;iter!=optimizedPoses.end();iter++)
    {
        node = nodes.find(iter->first)->second;
        node.sensorData().uncompressData();
        rtabmap::Transform trans2 = iter->second;
        cv::Mat next_image = node.sensorData().imageRaw();
        cv::Mat next_image1 = node.sensorData().rightRaw();
        pcl::IndicesPtr indices2(new std::vector<int>);
        std::vector<int> index2;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr sceneNext = rtabmap::util3d::cloudRGBFromSensorData(
                node.sensorData(),
                4,           // image decimation before creating the clouds
                20,        // maximum depth of the cloud
                0.0f,
                indices2.get());
        pcl::removeNaNFromPointCloud(*sceneNext, *sceneNext, index2);
        sceneNext = rtabmap::util3d::transformPointCloud(sceneNext,iter->second);

        transformation = sift_matching(prev_image, prev_image1, next_image, next_image1,
                                       trans1, trans2);
        if(transformation == identity)
            continue;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::transformPointCloud(*sceneNext, *transformed, transformation);

        if(!transformed->empty())
        {
            *Clouds += *transformed;
        }
        prev_image = next_image;
        prev_image1 = next_image1;
        trans1 = trans2;
    }

//    auto end = std::chrono::system_clock::now();
//    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
//    std::cout << "The matching costs " << double(duration.count())
//                 * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den
//              << " seconds" << std::endl;

    //Save final point cloud into .ply file
    pcl::PLYWriter writer;

    std::cout << "Saving cloud_match.ply... ("
              << static_cast<int>(Clouds->size()) << " points)" << std::endl;
    writer.write("sift/cloud_match.ply", *Clouds);
    std::cout << "Saving cloud_match.ply... done!" << std::endl;
//    for(size_t i = 0;i < optimizedPoses.size() / 2;i++)
//        iter++;
//    rtabmap::Signature node = nodes.find(iter->first)->second;
//    node.sensorData().uncompressData();
//    rtabmap::Transform trans1 = iter->second;
//    cv::Mat src = node.sensorData().imageRaw();
//    cv::Mat src1 = node.sensorData().rightRaw();
//    pcl::IndicesPtr indices1(new std::vector<int>);
//    std::vector<int> index1;
//    pcl::PointCloud<pcl::PointXYZRGB>::Ptr scenePrev = rtabmap::util3d::cloudRGBFromSensorData(
//            node.sensorData(),
//            4,           // image decimation before creating the clouds
//            20,        // maximum depth of the cloud
//            0.0f,
//            indices1.get());
//    pcl::removeNaNFromPointCloud(*scenePrev, *scenePrev, index1);
//    scenePrev = rtabmap::util3d::transformPointCloud(scenePrev,iter->second);

//    iter++;
//    node = nodes.find(iter->first)->second;
//    node.sensorData().uncompressData();
//    rtabmap::Transform trans2 = iter->second;
//    cv::Mat tgt = node.sensorData().imageRaw();
//    cv::Mat tgt1 = node.sensorData().rightRaw();
//    pcl::IndicesPtr indices2(new std::vector<int>);
//    std::vector<int> index2;
//    pcl::PointCloud<pcl::PointXYZRGB>::Ptr sceneNext = rtabmap::util3d::cloudRGBFromSensorData(
//            node.sensorData(),
//            4,           // image decimation before creating the clouds
//            20,        // maximum depth of the cloud
//            0.0f,
//            indices2.get());
//    pcl::removeNaNFromPointCloud(*sceneNext, *sceneNext, index2);
//    sceneNext = rtabmap::util3d::transformPointCloud(sceneNext,iter->second);
//    sift_matching(src, src1, tgt, tgt1, trans1, trans2, scenePrev, sceneNext);

    return 0;
}
