#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <chrono>
#include <ctime>
#include <climits>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

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

using namespace std;
using namespace g2o;
using namespace rtabmap;
using namespace cv;

struct Measurement
{
    Measurement(Eigen::Vector3d p, float g):pos_world(p), grayscale(g)
    {}
    Eigen::Vector3d pos_world;
    float grayscale;
};

inline Eigen::Vector3d project2Dto3D(int x, int y, float d, float fx, float fy,
                                     float cx, float cy, float scale)
{
    float zz = d / scale;
    float xx = zz * (x - cx) / fx;
    float yy = zz * (y - cy) / fy;
    return Eigen::Vector3d(xx, yy, zz);
}

inline Eigen::Vector2d project3Dto2D(float x, float y, float z, float fx, float fy, float cx, float cy)
{
    float u = fx * x / z + cx;
    float v = fy * y / z + cy;
    return Eigen::Vector2d(u, v);
}

bool poseEstimationDirect(const std::vector<Measurement> &measurements, cv::Mat* gray,
                          Eigen::Matrix3f &intrinsics, Eigen::Isometry3d &Tcw);

class EdgeSE3ProjectDirect: public g2o::BaseUnaryEdge< 1, double, g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeSE3ProjectDirect()
    {}

    EdgeSE3ProjectDirect(Eigen::Vector3d point, float fx, float fy, float cx, float cy, cv::Mat* image)
        :x_world_(point), fx_(fx), fy_(fy), cx_(cx), cy_(cy), image_(image)
    {}

    virtual void computeError()
    {
        const g2o::VertexSE3Expmap* v  = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
        Eigen::Vector3d x_local = v->estimate().map(x_world_);
        float x = x_local[0] * fx_ / x_local[2] + cx_;
        float y = x_local[1] * fy_ / x_local[2] + cy_;
        // check x,y is in the image
        if(x - 4 < 0 || (x + 4) > image_->cols || (y - 4) < 0 || (y + 4) > image_->rows)
        {
            _error(0, 0) = 0.0;
            this->setLevel(1);
        }
        else
        {
            _error(0, 0) = getPixelValue(x, y) - _measurement;
        }
    }

    // plus in manifold
    virtual void linearizeOplus()
    {
        if(level() == 1)
        {
            _jacobianOplusXi = Eigen::Matrix<double, 1, 6>::Zero();
            return;
        }
        g2o::VertexSE3Expmap* vtx = static_cast<g2o::VertexSE3Expmap*>(_vertices[0]);
        Eigen::Vector3d xyz_trans = vtx->estimate().map(x_world_);   // q in book

        double x = xyz_trans[0];
        double y = xyz_trans[1];
        double invz = 1.0 / xyz_trans[2];
        double invz_2 = invz * invz;

        float u = x * fx_ * invz + cx_;
        float v = y * fy_ * invz + cy_;

        // jacobian from se3 to u,v
        // NOTE that in g2o the Lie algebra is (\omega, \epsilon),
        //where \omega is so(3) and \epsilon the translation
        Eigen::Matrix<double, 2, 6> jacobian_uv_ksai;

        jacobian_uv_ksai(0, 0) = -x * y * invz_2 * fx_;
        jacobian_uv_ksai(0, 1) = (1 + (x * x * invz_2)) * fx_;
        jacobian_uv_ksai(0, 2) = - y * invz * fx_;
        jacobian_uv_ksai(0, 3) = invz * fx_;
        jacobian_uv_ksai(0, 4) = 0;
        jacobian_uv_ksai(0, 5) = -x * invz_2 * fx_;

        jacobian_uv_ksai(1, 0) = -(1 + y * y * invz_2) * fy_;
        jacobian_uv_ksai(1, 1) = x * y * invz_2 * fy_;
        jacobian_uv_ksai(1, 2) = x * invz * fy_;
        jacobian_uv_ksai(1, 3) = 0;
        jacobian_uv_ksai(1, 4) = invz * fy_;
        jacobian_uv_ksai(1, 5) = -y * invz_2 * fy_;

        Eigen::Matrix<double, 1, 2> jacobian_pixel_uv;

        jacobian_pixel_uv(0, 0) = (getPixelValue(u + 1, v) - getPixelValue(u - 1, v)) / 2;
        jacobian_pixel_uv(0, 1) = (getPixelValue(u, v + 1) - getPixelValue(u, v - 1)) / 2;

        _jacobianOplusXi = jacobian_pixel_uv * jacobian_uv_ksai;
    }

    // dummy read and write functions because we don't care...
    virtual bool read (std::istream& in)
    {}
    virtual bool write (std::ostream& out) const
    {}

protected:
    // get a gray scale value from reference image (bilinear interpolated)
    inline float getPixelValue(float x, float y)
    {
        uchar* data = &image_->data[int(y) * image_->step + int(x)];
        float xx = x - floor(x);
        float yy = y - floor(y);
        return float(
                   (1 - xx) * (1 - yy) * data[0] +
                   xx * (1 - yy) * data[1] +
                   (1 - xx) * yy * data[image_->step] +
                   xx * yy * data[image_->step + 1]);
    }
public:
    Eigen::Vector3d x_world_;   // 3D point in world frame
    float cx_ = 0, cy_ = 0, fx_ = 0, fy_ = 0; // Camera intrinsics
    cv::Mat* image_ = nullptr;    // reference image
};

int main ()
{
//    ULogger::setType(ULogger::kTypeConsole);
//    ULogger::setLevel(ULogger::kError);

//    bool isOptimate = 0;
//    bool isGlobal   = 1;

//    std::string dbPath = "/home/uisee/Documents/RTAB-Map/190715-091852.db";

//    // Get parameters
//    rtabmap::ParametersMap parameters;
//    rtabmap::DBDriver * driver = rtabmap::DBDriver::create();
//    if(driver->openConnection(dbPath))
//    {
//        parameters = driver->getLastParameters();
//        driver->closeConnection(false);
//    }
//    else
//    {
//        UERROR("Cannot open database %s!", dbPath.c_str());
//    }
//    delete driver;

//    // Get the global optimized map
//    rtabmap::Rtabmap rtabmap;
//    rtabmap.init(parameters, dbPath);

//    std::map<int, rtabmap::Signature> nodes;
//    std::map<int, rtabmap::Transform> optimizedPoses;
//    std::multimap<int, rtabmap::Link> links;
//    rtabmap.get3DMap(nodes, optimizedPoses, links, isOptimate, isGlobal);
//    std::map<int, rtabmap::Transform>::iterator iter=optimizedPoses.begin();
//    for(size_t i = 0;i < optimizedPoses.size() / 4;i++)
//        iter++;

//    rtabmap::Signature node = nodes.find(iter->first)->second;
//    node.sensorData().uncompressData();
//    rtabmap::StereoCameraModel model = node.sensorData().stereoCameraModel();

    cv::FileStorage fs("/home/uisee/Downloads/stereo_20190710_huadong_16.yaml", cv::FileStorage::READ);
    cv::Mat K_left, K_right, D_left, D_right, R_left, R_right, P_left, P_right;
    fs["M1"] >> K_left;
    fs["M2"] >> K_right;
    fs["D1"] >> D_left;
    fs["D2"] >> D_right;
    fs["R1"] >> R_left;
    fs["R2"] >> R_right;
    fs["P1"] >> P_left;
    fs["P2"] >> P_right;
    
    cv::Mat left, disparity, gray, right;
    std::string str1("/home/uisee/Data/image_capturer_0/1563771086.215176_00000002_206.tiff");
    std::string str2("/home/uisee/Data/image_capturer_0/1563771086.281687_00000003_208.tiff");
    cv::Mat img1 = cv::imread(str1);
    cv::Mat img2 = cv::imread(str2);
    auto str_img1 = str1.rfind(".tiff");
    auto str_img2 = str1.rfind(".tiff");
    std::string num1 = str1.substr(str_img1 - 8, 4);
    std::string num2 = str2.substr(str_img2 - 8, 4);
    cv::imwrite("1.png", img1);
    right = img1(cv::Rect(img1.cols / 2, 0, img1.cols / 2, img1.rows));
    left = img1(cv::Rect(0, 0, img1.cols / 2, img1.rows));
    cv::cvtColor(left, left, CV_BGR2GRAY);
    cv::cvtColor(right, right, CV_BGR2GRAY);

    cv::Mat map11, map12;
    cv::Mat map21, map22;
    cv::initUndistortRectifyMap(K_left, D_left, R_left, P_left, left.size(), CV_16SC2, map11, map12);
    cv::initUndistortRectifyMap(K_right, D_right, R_right, P_right, right.size(), CV_16SC2, map21, map22);
    cv::remap(left, left, map11, map12, INTER_LINEAR);
    cv::remap(right, right, map21, map22, INTER_LINEAR);
    cv::imwrite("left.png", left);
    disparity = rtabmap::util2d::disparityFromStereoImages(left, right);
    cv::imwrite("disparity.png", disparity);
    std::vector<Measurement> measurements;
    // 相机内参
    float cx = K_left.at<double>(0, 2);
    float cy = K_left.at<double>(1, 2);
    float fx = K_left.at<double>(0, 0);
    float fy = K_left.at<double>(1, 1);
    Eigen::Matrix3f K;
    K << K_left.at<double>(0, 0), K_left.at<double>(0, 1), K_left.at<double>(0, 2),
         K_left.at<double>(1, 0), K_left.at<double>(1, 1), K_left.at<double>(1, 2),
         K_left.at<double>(2, 0), K_left.at<double>(2, 1), K_left.at<double>(2, 2);

    Eigen::Isometry3d Tcw = Eigen::Isometry3d::Identity();

    cv::Mat prev_color;
    // 我们以第一个图像为参考，对后续图像和参考图像做直接法
    for(int x = 50;x < left.cols - 50;x++)
        for(int y = 50;y < left.rows - 50;y++)
        {
            Eigen::Vector2d delta (
                left.ptr<uchar>(y)[x + 1] - left.ptr<uchar>(y)[x - 1],
                left.ptr<uchar>(y + 1)[x] - left.ptr<uchar>(y - 1)[x]
            );
            if(delta.norm() < 50)
                continue;
            float disp = disparity.type() == CV_16SC1 ? float(disparity.at<short>(y, x)) / 16.0f
                                                      :disparity.at<float>(y, x);
            if(disp < 3)
                continue;

            float d = fx * 0.35 / disp;
            Eigen::Vector3d p3d = project2Dto3D(x, y, d, fx, fy, cx, cy, 1.0);
            //cv::Point3f point = util3d::projectDisparityTo3D(cv::Point2f(x, y), disp, model);
            //Eigen::Vector3d p3d(point.x, point.y, point.z);
            float grayscale = float(left.ptr<uchar>(y)[x]);
            measurements.push_back(Measurement(p3d, grayscale));
        }
    prev_color = left.clone();
    std::cout << "add total " << measurements.size() << " measurements." << std::endl;

    //iter++;

    //node = nodes.find(iter->first)->second;
    //node.sensorData().uncompressData();

    cv::Mat left1 = img2(cv::Rect(0, 0, img2.cols / 2, img2.rows));
    cv::Mat gray1;
    cv::cvtColor(left1, gray1, CV_BGR2GRAY);
    cv::remap(gray1, gray1, map11, map12, INTER_LINEAR);
    cv::imwrite("left2.png", gray1);

    // 使用直接法计算相机运动
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    poseEstimationDirect(measurements, &gray1, K, Tcw);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    std::cout << "direct method costs time: " << time_used.count() << " seconds." << std::endl;
    std::cout << "Tcw = " << Tcw.matrix() << std::endl;

    // plot the feature points
    cv::Mat img_show(left.rows * 2, left.cols, CV_8UC1);
    prev_color.copyTo(img_show(cv::Rect(0, 0, left.cols, left.rows)));
    gray1.copyTo(img_show(cv::Rect(0, left.rows, left.cols, left.rows)));
    cv::cvtColor(img_show, img_show, CV_GRAY2BGR);
    for(Measurement m:measurements)
    {
        if(rand() > RAND_MAX/5)
            continue;
        Eigen::Vector3d p = m.pos_world;
        Eigen::Vector2d pixel_prev = project3Dto2D(p(0, 0), p(1, 0), p(2, 0), fx, fy, cx, cy);
        Eigen::Vector3d p2 = Tcw * m.pos_world;
        Eigen::Vector2d pixel_now = project3Dto2D(p2(0, 0), p2(1, 0), p2(2, 0), fx, fy, cx, cy);
        if(pixel_now(0, 0) < 0 || pixel_now(0, 0) >= left.cols || pixel_now(1, 0) < 0
                || pixel_now(1, 0) >= left.rows)
            continue;

        float b = 0;
        float g = 250;
        float r = 0;
        img_show.ptr<uchar>(pixel_prev(1, 0))[int(pixel_prev(0, 0)) * 3] = b;
        img_show.ptr<uchar>(pixel_prev(1, 0))[int(pixel_prev(0, 0)) * 3 + 1] = g;
        img_show.ptr<uchar>(pixel_prev(1, 0))[int(pixel_prev(0, 0)) * 3 + 2] = r;

        img_show.ptr<uchar>(pixel_now(1, 0) + left.rows)[int(pixel_now(0, 0)) * 3] = b;
        img_show.ptr<uchar>(pixel_now(1, 0) + left.rows)[int(pixel_now(0, 0)) * 3 + 1] = g;
        img_show.ptr<uchar>(pixel_now(1, 0) + left.rows)[int(pixel_now(0, 0)) * 3 + 2] = r;
        cv::circle(img_show, cv::Point2d(pixel_prev(0, 0), pixel_prev(1, 0)), 4, cv::Scalar(b, g, r), 2);
        cv::circle(img_show, cv::Point2d(pixel_now(0, 0), pixel_now(1, 0) + left.rows), 4,
                   cv::Scalar(b, g, r), 2);
    }
    cv::imwrite("result" + num1 + "_" + num2 + ".png", img_show);

    cv::Mat R,R1, R2, P1, P2, Q;
    cv::Mat K_lef = (cv::Mat_<double>(3,3)<< 762.72, 0.0, 640.0, 0.0, 762.72, 360.0, 0.0, 0.0, 1.0);
    cv::Mat K_righ = (cv::Mat_<double>(3,3)<< 762.72, 0.0, 640.0, 0.0, 762.72, 360.0, 0.0, 0.0, 1.0);
    cv::Mat D_lef = (cv::Mat_<double>(1,5) << 0.0, 0.0, 0.0, 0.0, 0.0);
    cv::Mat D_righ = (cv::Mat_<double>(1,5)<< 0.0, 0.0, 0.0, 0.0, 0.0);

    R = (cv::Mat_<double>(3,3)<< 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    cv::Mat t = (cv::Mat_<double>(3,1) << 0.35,0.0,0.0);
    cv::stereoRectify(K_lef, D_lef, K_righ, D_righ,cv::Size(1280, 720), R, t,R1, R2, P1, P2, Q);
    cout << P1  <<  P2<< endl;


    return 0;
}

bool poseEstimationDirect(const vector<Measurement>& measurements, cv::Mat* gray,
                          Eigen::Matrix3f& K, Eigen::Isometry3d& Tcw)
{
    // 初始化g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,1>> DirectBlock;  // 求解的向量是6＊1的
    DirectBlock::LinearSolverType* linearSolver = new g2o::LinearSolverDense<DirectBlock::PoseMatrixType>();
    DirectBlock* solver_ptr = new DirectBlock(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
    pose->setEstimate(g2o::SE3Quat(Tcw.rotation(), Tcw.translation()));
    pose->setId(0);
    optimizer.addVertex(pose);

    // 添加边
    int id = 1;
    for(Measurement m : measurements)
    {
        EdgeSE3ProjectDirect* edge = new EdgeSE3ProjectDirect (
            m.pos_world,
            K(0, 0), K(1, 1), K(0, 2), K(1, 2), gray
        );
        edge->setVertex(0, pose);
        edge->setMeasurement(m.grayscale);
        edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
        edge->setId(id++);
        optimizer.addEdge(edge);
    }
    std::cout << "edges in graph: " << optimizer.edges().size() << std::endl;
    optimizer.initializeOptimization();
    optimizer.optimize(50);
    Tcw = pose->estimate();
}
