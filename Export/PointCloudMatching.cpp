/*
Copyright (c) 2010-2016, Mathieu Labbe - IntRoLab - Universite de Sherbrooke
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Universite de Sherbrooke nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

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
#include "pcl/point_types.h"
#include "pcl/registration/icp.h"
#include "pcl/io/ply_io.h"
#include "pcl/io/obj_io.h"
#include "pcl/common/common.h"
#include "pcl/surface/poisson.h"
#include <stdio.h>
#include <chrono>
#include <sstream>

using namespace rtabmap;

void showUsage()
{
    std::cout << std::endl
              << "Usage:" << std::endl
              << "rtabmap-matching database.db" << std::endl << std::endl;
    exit(1);
}

int main(int argc, char * argv[])
{
    ULogger::setType(ULogger::kTypeConsole);
    ULogger::setLevel(ULogger::kError);

    if(argc < 2)
    {
        showUsage();
    }

    bool isOptimate = 0;
    bool isGlobal   = 1;
    int maxDistance = 20.f;

    std::cout << "isOptimate:" << isOptimate << std::endl;
    std::cout << "isGlobal:"   << isGlobal << std::endl;
    std::cout << "maxDistance:"<< maxDistance << std::endl;

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

    //Count the time
    auto start = std::chrono::system_clock::now();
    std::cout << "Start matching point cloud......" << std::endl;

    // Get the global optimized map
    Rtabmap rtabmap;
    rtabmap.init(parameters, dbPath);

    std::map<int, Signature> nodes;
    std::map<int, Transform> optimizedPoses;
    std::multimap<int, Link> links;
    rtabmap.get3DMap(nodes, optimizedPoses, links, isOptimate, isGlobal);

    // Construct the cloud
    std::vector<int> index;
    std::map<int, Transform>::iterator iter = optimizedPoses.begin();
    Signature node = nodes.find(iter->first)->second;

    node.sensorData().uncompressData();
    //std::vector<CameraModel> models = node.sensorData().cameraModels();
    //cv::Mat depth = node.sensorData().depthRaw();
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr Clouds(new pcl::PointCloud<pcl::PointXYZRGB>);

    pcl::IndicesPtr indices(new std::vector<int>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud = util3d::cloudRGBFromSensorData(
            node.sensorData(),
            4,           // image decimation before creating the clouds
            maxDistance,        // maximum depth of the cloud
            0.0f,
            indices.get());

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformedCloud_tgt(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::removeNaNFromPointCloud(*cloud,*transformedCloud_tgt,index);
    transformedCloud_tgt = util3d::transformPointCloud(transformedCloud_tgt,iter->second);
    iter++;

    pcl::IterativeClosestPoint<pcl::PointXYZRGB,pcl::PointXYZRGB> icp;
    double maxCorDis = 0.5;
    double tranEps = 1e-8;
    double EucFitEps = 0.01;
    icp.setMaxCorrespondenceDistance(maxCorDis);
    icp.setMaximumIterations(50);
    icp.setTransformationEpsilon(tranEps);
    icp.setEuclideanFitnessEpsilon(EucFitEps);

    //Iteratively match each frame and add into point cloud
    for(unsigned int i = 1; i < optimizedPoses.size() / 2; ++i)
    {
        pcl::PointCloud<pcl::PointXYZRGB> final;
        node = nodes.find(iter->first)->second;

        node.sensorData().uncompressData();
        //models = node.sensorData().cameraModels();

        cloud = util3d::cloudRGBFromSensorData(
                node.sensorData(),
                4,           // image decimation before creating the clouds
                maxDistance,        // maximum depth of the cloud
                0.0f,
                indices.get());

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformedCloud_src(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::removeNaNFromPointCloud(*cloud,*transformedCloud_src,index);
        transformedCloud_src = util3d::transformPointCloud(transformedCloud_src,iter->second);

        icp.setInputSource(transformedCloud_src);
        icp.setInputTarget(transformedCloud_tgt);
        icp.align(final);

        if(!final.empty())
        {
            *Clouds += final;
        }

        //auto mat=icp.getFinalTransformation();

        //This frame's matching result will be target of next frame
        transformedCloud_tgt->clear();
        *transformedCloud_tgt = final;
        cloud->clear();
        iter++;
    }

    auto end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "The matching costs " << double(duration.count())
                 * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den
              << " seconds" << std::endl;

    //Save final point cloud into .ply file
    pcl::PLYWriter writer;
    std::stringstream ss;
    ss << maxCorDis << "_" << tranEps << "_" << EucFitEps;
    std::string s = ss.str();

    std::cout << "Saving cloud_match_" + date + "_" + s + "_half.ply... ("
              << static_cast<int>(Clouds->size()) << " points)" << std::endl;
    writer.write(date + "/cloud_match_" + date + "_" + s + "_half.ply", *Clouds);
    std::cout << "Saving cloud_match_" + date + "_" + s + "_half.ply... done!" << std::endl;

    return 0;
}
