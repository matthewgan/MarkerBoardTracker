#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <sstream>
#include <fstream>

using namespace cv;
using namespace std;

int startWebCamMonitoring(const Mat& cameraMatrix, const Mat& distanceCoefficients, float arucoSquareDimension)
{
    Mat frame;

    VideoCapture vid(2);

    if(!vid.isOpened())
    {
        return -1;
    }

    namedWindow("webcam", CV_WINDOW_AUTOSIZE);

    while(true)
    {
        if(!vid.read(frame))
        {
            break;
        }

        flip(frame, frame, 1);

        imshow("webcam", frame);

        if(waitKey(30) >= 0) break;
    }

    return 1;
}

int main(int argc, char const *argv[])
{
    Mat cameraMatrix = Mat::eye(3,3,CV_64F);

    Mat distanceCoefficients;

    startWebCamMonitoring(cameraMatrix, distanceCoefficients, 0.01f);

    return 0;
}
