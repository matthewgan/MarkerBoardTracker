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

const float calibrationSquareDimension = 0.02305f;
const float arucoSquareDimension = 0.1016f;
const Size chessboardDimension = Size(6,9);

void createArucoMarkers()
{
    Mat outputMarker;

    Ptr<aruco::Dictionary> markerDictionary = aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME::DICT_4X4_50);

    for(int i=0;i<50;i++)
    {
        aruco::drawMarker(markerDictionary, i, 500, outputMarker, 1);
        ostringstream convert;
        string imageName = "4x4Marker_";
        convert << imageName << i << ".jpg";
        imwrite(convert.str(), outputMarker);
    }
}

void createKnownBoardPosition(Size boardSize, float squareEdgeLength, vector<Point3f>& corners)
{
    for(int i=0;i<boardSize.height; i++)
    {
        for(int j=0;j<boardSize.width;j++)
        {
            corners.push_back(Point3f(j*squareEdgeLength, i*squareEdgeLength, 0.0f));
        }
    }
}

void getChessboardCorners(vector<Mat> images, vector<vector<Point2f>>& allFoundCorners, bool showResult=false)
{
    for(vector<Mat>::iterator iter = images.begin(); iter!=images.end();iter++)
    {
        vector<Point2f> pointBuf;
        bool found = findChessboardCorners(*iter, Size(9,6), pointBuf, CV_CALIB_CB_ADAPTIVE_THRESH|CV_CALIB_CB_NORMALIZE_IMAGE);

        if(found)
        {
            allFoundCorners.push_back(pointBuf);
        }

        if(showResult)
        {
            drawChessboardCorners(*iter, Size(9,6), pointBuf, found);
            imshow("Looking for corners", *iter);
            waitKey(0);
        }
    }
}

void cameraCalibration(vector<Mat> calibrationImages, Size boardSize, float squareEdgeLength, Mat& cameraMatrix, Mat& distanceCoefficients)
{
    vector<vector<Point2f>> checkBoardImagesSpacePoints;
    getChessboardCorners(calibrationImages, checkBoardImagesSpacePoints, false);

    vector<vector<Point3f>> worldSpaceCornerPoints(1);

    createKnownBoardPosition(boardSize, squareEdgeLength, worldSpaceCornerPoints[0]);
    worldSpaceCornerPoints.resize(checkBoardImagesSpacePoints.size(), worldSpaceCornerPoints[0]);

    vector<Mat> rVectors, tVectors;
    distanceCoefficients = Mat::zeros(8,1,CV_64F);

    calibrateCamera(worldSpaceCornerPoints, checkBoardImagesSpacePoints, boardSize, cameraMatrix, distanceCoefficients, rVectors, tVectors);


}

bool saveCameraCalibration(string name, Mat cameraMatrix, Mat distanceCoefficients)
{
    ofstream outStream(name);
    if(outStream)
    {
        uint16_t rows = cameraMatrix.rows;
        uint16_t columns = cameraMatrix.cols;

        for(int r=0;r<rows;r++)
        {
            for(int c=0;c<columns;c++)
            {
                double value = cameraMatrix.at<double>(r,c);
                outStream << value << endl;
            }
        }

        rows = distanceCoefficients.rows;
        columns = distanceCoefficients.cols;

        for(int r=0;r<rows;r++)
        {
            for(int c=0;c<columns;c++)
            {
                double value = distanceCoefficients.at<double>(r,c);
                outStream << value << endl;
            }
        }

        outStream.close();
        return true;
    }
    return false;
}

int startWebCamMonitoring(const Mat& cameraMatrix, const Mat& distanceCoefficients, float arucoSquareDimension)
{
    Mat frame;
    
    vector<int> markerIds;

    vector<vector<Point2f>> markerCorners, rejectedCandidates;

    aruco::DetectorParameters parameters;

    Ptr<aruco::Dictionary> markerDictionary = aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME::DICT_4X4_50);

    VideoCapture vid(2);

    if(!vid.isOpened())
    {
        return -1;
    }

    namedWindow("webcam", CV_WINDOW_AUTOSIZE);

    vector<Vec3d> rotaionVectors, translationVectors;

    while(true)
    {
        if(!vid.read(frame))
        {
            break;
        }

        aruco::detectMarkers(frame, markerDictionary, markerCorners, markerIds);

        aruco::estimatePoseSingleMarkers(markerCorners, arucoSquareDimension, cameraMatrix, distanceCoefficients, rotaionVectors, translationVectors);

        for(int i=0;i<markerIds.size();i++)
        {
            aruco::drawAxis(frame, cameraMatrix, distanceCoefficients, rotaionVectors, translationVectors, 0.1f);
        }

        imshow("webcam", frame);

        if(waitKey(30) >= 0) break;
    }

    return 1;
}

int main(int argc, char const *argv[])
{
    Mat frame;
    Mat drawToFrame;

    Mat cameraMatrix = Mat::eye(3,3,CV_64F);

    Mat distanceCoefficients;

    vector<Mat> savedImages;

    vector<vector<Point2f>> markerCorners, rejectedCandidates;

    VideoCapture vid(2);

    if(!vid.isOpened())
    {
        return 0;
    }

    int framesPerSecond = 20;

    namedWindow("webcam", CV_WINDOW_AUTOSIZE);

    while(true)
    {
        if(!vid.read(frame))
        {
            break;
        }

        vector<Vec2f> foundPoints;
        bool found = false;

        found = findChessboardCorners(frame, chessboardDimension, foundPoints, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);
        frame.copyTo(drawToFrame);
        drawChessboardCorners(drawToFrame, chessboardDimension, foundPoints, found);
        if(found)
        {
            imshow("webcam", drawToFrame);
        }
        else
        {
            imshow("webcam", frame);
        }
        char character = waitKey(1000/framesPerSecond);

        switch (character)
        {
        case ' ':
            //saving image
            if(found)
            {
                Mat temp;
                frame.copyTo(temp);
                savedImages.push_back(temp);
            }
            break;

        case 13:
            //start calibration
            if(savedImages.size() > 15)
            {
                cameraCalibration(savedImages, chessboardDimension, calibrationSquareDimension, cameraMatrix, distanceCoefficients);
                saveCameraCalibration("cameraCalibration", cameraMatrix, distanceCoefficients);
            }
            break;

        case 27:
            //exit
            break;
        
        default:
            break;
        }

    }

    return 0;
}
