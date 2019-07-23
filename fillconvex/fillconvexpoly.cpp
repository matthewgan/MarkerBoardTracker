#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

int main(int argc, char const *argv[])
{
    Mat image(50, 50, CV_8UC3, Scalar(0,0,0));
    vector<Point> pts;
    pts.push_back(Point(5, 10));    //top-left
    pts.push_back(Point(45, 5));    //top-right
    pts.push_back(Point(45, 45));   //bottom-right
    pts.push_back(Point(5, 40));    //bottom-left

    fillConvexPoly(image, pts, Scalar(255,255,255));

    imshow("img", image);

    while(true)
    {
        if(waitKey(30) >= 0) break;
    }




    return 0;
}
