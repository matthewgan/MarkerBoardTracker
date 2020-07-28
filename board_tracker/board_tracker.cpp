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

const float calibrationSquareDimension = 0.024f;
const float arucoSquareDimension = 0.042f;
const Size chessboardDimension = Size(6,9);

bool loadCameraCalibration(string name, Mat& cameraMatrix, Mat& distanceCoefficients)
{
    ifstream inStream(name);
    if(inStream)
    {
        uint16_t rows;
        uint16_t columns;

        inStream >> rows;
        inStream >> columns;

        cameraMatrix = Mat{Size(columns, rows), CV_64F};

        for(int r=0;r<rows;r++)
        {
            for(int c=0;c<columns;c++)
            {
                double read = 0.0f;
                inStream >> read;
                cameraMatrix.at<double>(r,c) = read;
                cout << cameraMatrix.at<double>(r,c) << endl;
            }
        }

        inStream >> rows;
        inStream >> columns;

        distanceCoefficients = Mat::zeros(rows, columns, CV_64F);

        for(int r=0;r<rows;r++)
        {
            for(int c=0;c<columns;c++)
            {
                double read = 0.0f;
                inStream >> read;
                distanceCoefficients.at<double>(r,c) = read;
                cout << distanceCoefficients.at<double>(r,c) << endl;
            }
        }

        inStream.close();
        return true;
    }
    return false;
}

static bool readDetectorParameters(string filename, Ptr<aruco::DetectorParameters> &params) {
    FileStorage fs(filename, FileStorage::READ);
    if(!fs.isOpened())
        return false;
    fs["adaptiveThreshWinSizeMin"] >> params->adaptiveThreshWinSizeMin;
    fs["adaptiveThreshWinSizeMax"] >> params->adaptiveThreshWinSizeMax;
    fs["adaptiveThreshWinSizeStep"] >> params->adaptiveThreshWinSizeStep;
    fs["adaptiveThreshConstant"] >> params->adaptiveThreshConstant;
    fs["minMarkerPerimeterRate"] >> params->minMarkerPerimeterRate;
    fs["maxMarkerPerimeterRate"] >> params->maxMarkerPerimeterRate;
    fs["polygonalApproxAccuracyRate"] >> params->polygonalApproxAccuracyRate;
    fs["minCornerDistanceRate"] >> params->minCornerDistanceRate;
    fs["minDistanceToBorder"] >> params->minDistanceToBorder;
    fs["minMarkerDistanceRate"] >> params->minMarkerDistanceRate;
    fs["cornerRefinementMethod"] >> params->cornerRefinementMethod;
    fs["cornerRefinementWinSize"] >> params->cornerRefinementWinSize;
    fs["cornerRefinementMaxIterations"] >> params->cornerRefinementMaxIterations;
    fs["cornerRefinementMinAccuracy"] >> params->cornerRefinementMinAccuracy;
    fs["markerBorderBits"] >> params->markerBorderBits;
    fs["perspectiveRemovePixelPerCell"] >> params->perspectiveRemovePixelPerCell;
    fs["perspectiveRemoveIgnoredMarginPerCell"] >> params->perspectiveRemoveIgnoredMarginPerCell;
    fs["maxErroneousBitsInBorderRate"] >> params->maxErroneousBitsInBorderRate;
    fs["minOtsuStdDev"] >> params->minOtsuStdDev;
    fs["errorCorrectionRate"] >> params->errorCorrectionRate;
    return true;
}

int main(int argc, char const *argv[])
{
    int markersX = 4;
    int markersY = 6;
    float markerLength = 0.04f;
    float markerSeparation = 0.002f;
    int dictionaryId = aruco::PREDEFINED_DICTIONARY_NAME::DICT_6X6_50;
    bool showRejected = false;
    bool refindStrategy = true;
    int camId = 2;

    Mat cameraMatrix = Mat::eye(3,3,CV_64F);
    Mat distanceCoefficients;    
    loadCameraCalibration("cameraCalibration", cameraMatrix, distanceCoefficients);

    Ptr<aruco::DetectorParameters> detectorParams = aruco::DetectorParameters::create();
    
    bool readOk = readDetectorParameters("detector_params.yml", detectorParams);
    if(!readOk) {
        cerr << "Invalid detector parameters file" << endl;
        return 0;
    }

    detectorParams->cornerRefinementMethod = aruco::CORNER_REFINE_SUBPIX;   

    Ptr<aruco::Dictionary> dictionary =
        aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));

    float axisLength = 0.5f * ((float)min(markersX, markersY) * (markerLength + markerSeparation) +
                               markerSeparation);

    // create board object
    Ptr<aruco::GridBoard> gridboard = aruco::GridBoard::create(markersX, markersY, markerLength, markerSeparation, dictionary);

    cout << "created board" << endl;

    Mat frame;

    Mat frameCopy;
    
    vector<int> markerIds;

    vector<vector<Point2f>> markerCorners, rejectedCandidates;

    VideoCapture vid(0);

    if(!vid.isOpened())
    {
        return -1;
    }

    //namedWindow("before", CV_WINDOW_AUTOSIZE);

    //namedWindow("after", CV_WINDOW_AUTOSIZE);

    namedWindow("before");

    namedWindow("after");

    vector<Vec3d> rotaionVectors, translationVectors;

    Vec3d rvec, tvec;

    while(true)
    {
        if(!vid.read(frame))
        {
            break;
        }

        aruco::detectMarkers(frame, dictionary, markerCorners, markerIds, detectorParams, rejectedCandidates);

        if(refindStrategy)
        {
            aruco::refineDetectedMarkers(frame, gridboard, markerCorners, markerIds, rejectedCandidates, cameraMatrix, distanceCoefficients);
        }

        //cout << markerIds.size() << endl;

        // for(int i=0;i<markerIds.size();i++)
        // {
        //     aruco::drawDetectedMarkers(frame, markerCorners, markerIds);
        //     // aruco::drawAxis(frame, cameraMatrix, distanceCoefficients, 
        //     //     rotaionVectors[i], translationVectors[i], 0.1f);
        // }

        int markerOfBoardDetected = 0;
        if(markerIds.size()>0)
        {
            markerOfBoardDetected = aruco::estimatePoseBoard(markerCorners, markerIds, gridboard, cameraMatrix, distanceCoefficients,
                rvec, tvec);
        }

        //cout << markerOfBoardDetected << endl;

        imshow("before", frame);

        frame.copyTo(frameCopy);

        if(markerIds.size()>0)
        {
            aruco::drawDetectedMarkers(frameCopy, markerCorners, markerIds);
        }

        if(showRejected && rejectedCandidates.size() > 0)
        {
            aruco::drawDetectedMarkers(frameCopy, rejectedCandidates, noArray(), Scalar(100, 0, 255));
        }

        if(markerOfBoardDetected > 0)
        {
            aruco::drawAxis(frameCopy, cameraMatrix, distanceCoefficients, rvec, tvec, axisLength);
        }

        imshow("after", frameCopy);

        if(waitKey(30) >= 0) break;
    }

    return 1;

    // VideoCapture vid(camId);

    // if(!vid.isOpened())
    // {
    //     return -1;
    // }

    // Mat frame, frameCopy;
    
    // vector<int> markerIds;

    // vector<vector<Point2f>> markerCorners, rejectedCandidates;

    // Ptr<aruco::Dictionary> markerDictionary = aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME::DICT_6X6_50);

    // vector<Vec3d> rotaionVectors, translationVectors;

    // namedWindow("webcam", CV_WINDOW_AUTOSIZE);

    // while (true)
    // {
    //     if(!vid.read(frame))
    //     {
    //         break;
    //     }

    //     aruco::detectMarkers(frame, markerDictionary, markerCorners, markerIds, detectorParams, rejectedCandidates, cameraMatrix, distanceCoefficients);

    //     if (refindStrategy)
    //     {
    //         aruco::refineDetectedMarkers(frame, gridboard, markerCorners, markerIds, rejectedCandidates, cameraMatrix, distanceCoefficients);
    //     }

    //     int markersOfBoardDetected = 0;
    //     if(markerIds.size() > 0)
    //     {
    //         markersOfBoardDetected = aruco::estimatePoseBoard(markerCorners, markerIds, gridboard, cameraMatrix, distanceCoefficients, rotaionVectors, translationVectors);
    //     }

    //     frame.copyTo(frameCopy);

    //     if(markerIds.size()>0)
    //     {
    //         aruco::drawDetectedMarkers(frameCopy, markerCorners, markerIds);
    //     }

    //     if(showRejected && rejectedCandidates.size()>0)
    //     {
    //         aruco::drawDetectedMarkers(frameCopy, rejectedCandidates, noArray(), Scalar(100,0,255));
    //     }

    //     if(markersOfBoardDetected>0)
    //     {
    //         aruco::drawAxis(frameCopy, cameraMatrix, distanceCoefficients, rotaionVectors, translationVectors, axisLength);
    //     }

    //     imshow("webcam", frameCopy);

    //     char key = (char)waitKey(1);
    //     if(key == 27) break;
    // }

    return 0;
}
