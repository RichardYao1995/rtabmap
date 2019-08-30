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
#include "rtabmap/core/util2d.h"
#include "rtabmap/utilite/UMath.h"
#include "rtabmap/utilite/UTimer.h"
#include "rtabmap/utilite/UFile.h"
#include "pcl/filters/filter.h"
#include "pcl/io/ply_io.h"
#include "pcl/io/obj_io.h"
#include "pcl/common/common.h"
#include "pcl/surface/poisson.h"
#include "include/StereoEfficientLargeScale.h"
#include <stdio.h>
#include <chrono>

using namespace rtabmap;
using namespace std;

void showUsage()
{
    printf("\nUsage:\n"
            "rtabmap-exportCloud [options] database.db\n"
            "Options:\n"
            "    --mesh          Create a mesh.\n"
            "    --texture       Create a mesh with texture.\n"
            "\n");
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
    //float oneVoxelize = 0.02f;
    //float allVoxelize = 0.1f;

    std::cout<< "isOptimate:" <<isOptimate<<std::endl;
    std::cout<< "isGlobal:"   <<isGlobal<<std::endl;
    std::cout<< "maxDistance:"<<maxDistance<<std::endl;


//    bool mesh = false;
//    bool texture = false;
//    for(int i=1; i<argc-1; ++i)
//    {
//        if(std::strcmp(argv[i], "--mesh") == 0)
//        {
//            mesh = true;
//        }
//        else if(std::strcmp(argv[i], "--texture") == 0)
//        {
//            texture = true;
//        }
//    }

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

    auto start = std::chrono::system_clock::now();
    std::cout << "Start exporting point cloud......" << std::endl;

    // Get the global optimized map
    Rtabmap rtabmap;
    rtabmap.init(parameters, dbPath);

    std::map<int, Signature> nodes;
    std::map<int, Transform> optimizedPoses;
    std::multimap<int, Link> links;
    rtabmap.get3DMap(nodes, optimizedPoses, links, isOptimate, isGlobal);


    // Construct the cloud
    //pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr mergedClouds(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    //std::map<int, rtabmap::Transform> cameraPoses;
    //std::map<int, std::vector<rtabmap::CameraModel> > cameraModels;
    //std::map<int, cv::Mat> cameraDepths;
    //int i(0);
    std::cout << 0 << std::endl;
    std::vector<int> index;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr Clouds(new pcl::PointCloud<pcl::PointXYZRGB>);
    std::map<int, Transform>::iterator iter=optimizedPoses.begin();
    for(unsigned int i = 0; i < optimizedPoses.size(); ++i)
    {
        Signature node = nodes.find(iter->first)->second;

        // uncompress data
        node.sensorData().uncompressData();
        //std::vector<CameraModel> models = node.sensorData().cameraModels();
        //cv::Mat depth = node.sensorData().depthRaw();

//        pcl::IndicesPtr indices(new std::vector<int>);
//        cv::Mat color = node.sensorData().imageRaw();
//        cv::Mat right = node.sensorData().rightRaw();
//        cv::Mat left;
//        cv::cvtColor(color, left, CV_BGR2GRAY);

//        StereoEfficientLargeScale elas(0,128);
//        cv::Mat dest, result;
//        elas(left,right,dest,100);
//        dest.convertTo(result,CV_32FC1,1.0/16);

//        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudFromDisparityRGB = util3d::cloudFromDisparityRGB(
//                    color, result, node.sensorData().stereoCameraModel(), 4, maxDistance, 0.0f, indices.get());

//        if(cloudFromDisparityRGB->size() && !node.sensorData().stereoCameraModel().left().localTransform().isNull()
//                && !node.sensorData().stereoCameraModel().left().localTransform().isIdentity())
//        {
//            cloudFromDisparityRGB = util3d::transformPointCloud(cloudFromDisparityRGB,
//                                                                node.sensorData().stereoCameraModel().left().localTransform());
//        }
//        cout << iter->second << endl;

        pcl::IndicesPtr indices(new std::vector<int>);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud = util3d::cloudRGBFromSensorData(
                node.sensorData(),
                4,           // image decimation before creating the clouds
                maxDistance,        // maximum depth of the cloud
                0.0f,
                indices.get());

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformedCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::removeNaNFromPointCloud(*cloud,*transformedCloud,index);
        if(!transformedCloud->empty())
        {
            *Clouds+=*util3d::transformPointCloud(transformedCloud,iter->second);
        }
        iter++;
    }
//        if(maxDistance == 20)
//            transformedCloud = rtabmap::util3d::voxelize(cloud, indices, oneVoxelize);
//        else
//            transformedCloud = rtabmap::util3d::voxelize(cloud, indices, 0.01f);

//        transformedCloud = rtabmap::util3d::transformPointCloud(transformedCloud, iter->second);

//        Eigen::Vector3f viewpoint( iter->second.x(),  iter->second.y(),  iter->second.z());
//        pcl::PointCloud<pcl::Normal>::Ptr normals = rtabmap::util3d::computeNormals(transformedCloud, 10, 0.0f, viewpoint);

//        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloudWithNormals(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
//        pcl::concatenateFields(*transformedCloud, *normals, *cloudWithNormals);

//        if(mergedClouds->size() == 0)
//        {
//            *mergedClouds = *cloudWithNormals;
//        }
//        else
//        {
//            *mergedClouds += *cloudWithNormals;
//        }

//        cameraPoses.insert(std::make_pair(iter->first, iter->second));
//        if(!models.empty())
//        {
//            cameraModels.insert(std::make_pair(iter->first, models));
//        }
//        if(!depth.empty())
//        {
//            cameraDepths.insert(std::make_pair(iter->first, depth));
//        }

//        ++i;
////        if(i > (int)optimizedPoses.size()/2)
////            break;
//    }
//    if(mergedClouds->size())
//    {
//        if(!(mesh || texture))
//        {
//            //std::cout<<"mergedClouds : "<<std::endl;
//            printf("Voxel grid filtering of the assembled cloud (voxel=%f, %d points)\n", 0.01f, static_cast<int>(mergedClouds->size()));
//            if(maxDistance != 20)
//                mergedClouds = util3d::voxelize(mergedClouds, 0.01f);
//            else
//                mergedClouds = util3d::voxelize(mergedClouds, allVoxelize);

    auto end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "The exporting costs " << double(duration.count())
                 * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den
              << " seconds" << std::endl;

    std::cout << "Saving cloud_origin_" + date + "_noslam.ply... (" << static_cast<int>(Clouds->size()) << " points)" << std::endl;
    pcl::PLYWriter writer;
    writer.write(date + "/cloud_origin_" + date + "_noslam.ply", *Clouds);
    std::cout << "Saving cloud_origin_" + date + "_noslam.ply... done!" << std::endl;
//        }
//    }
//    else
//    {
//        printf("Export failed! The cloud is empty.\n");
//    }

    return 0;
}
