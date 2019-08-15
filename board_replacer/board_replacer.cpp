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
                // cout << cameraMatrix.at<double>(r,c) << endl;
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
                // cout << distanceCoefficients.at<double>(r,c) << endl;
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

static bool readMarkerBoardParameters(string filename)
{

}

void order_points(vector<Point2f> &pts)
{
    sort(pts.begin(), pts.end(), [](Point2f a, Point2f b) {
        return a.x < b.x;
    });

 

    if (pts[0].y > pts[1].y)
    {
        swap(pts[0], pts[1]);
    }
    if (pts[2].y < pts[3].y)
    {
        swap(pts[2], pts[3]);
    }
    swap(pts[1], pts[3]);
}

namespace {
const char* about = "replace board with one image";
const char* keys  = "{@outfile |<none> | Output image }";
}

int main(int argc, char const *argv[])
{
    CommandLineParser parser(argc, argv, keys);
    parser.about(about);

    if(argc < 1) {
        parser.printMessage();
        return 0;
    }

    String img_filename = parser.get<String>(0);

    if(!parser.check()) {
        parser.printErrors();
        return 0;
    }

    try{
    int markersX = 4;
    int markersY = 6;
    float markerLength = 0.04f;
    float markerSeparation = 0.002f;
    int dictionaryId = aruco::PREDEFINED_DICTIONARY_NAME::DICT_6X6_50;
    bool showRejected = false;
    bool refindStrategy = true;
    int camId = 2;
    // string img_filename = "2.png";

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

    Mat logo = imread(img_filename);

    // add image on the marker
    namedWindow("logo", CV_WINDOW_AUTOSIZE);

    namedWindow("before", CV_WINDOW_AUTOSIZE);

    namedWindow("after", CV_WINDOW_AUTOSIZE);

    namedWindow("mask", CV_WINDOW_AUTOSIZE);

    imshow("logo", logo);

    // calculate the logo alpha
    float logo_density = mean(mean(logo))[0];

    Point2f inputQuad[4];
    Point2f outputQuad[4];

    //get 4 points from logo, from top-left in clockwise order
    inputQuad[0] = Point2f(0,0);
    inputQuad[1] = Point2f(logo.cols-1, 0);
    inputQuad[2] = Point2f(logo.cols-1, logo.rows-1);
    inputQuad[3] = Point2f(0, logo.rows-1);

    Mat frame;

    Mat frameCopy;

    Mat blended;
    
    vector<int> markerIds;

    vector<vector<Point2f>> markerCorners, rejectedCandidates;

    VideoCapture vid(camId);

    if(!vid.isOpened())
    {
        return -1;
    }    

    vector<Vec3d> rotaionVectors, translationVectors;

    Vec3d rvec, tvec;

    cout << "ready to loop" << endl;    

    while(true)
    {
        if(!vid.read(frame))
        {
            break;
        } 

        aruco::detectMarkers(frame, dictionary, markerCorners, markerIds, detectorParams, rejectedCandidates);

        if((refindStrategy) && (markerIds.size()>0))
        {
            aruco::refineDetectedMarkers(frame, gridboard, markerCorners, markerIds, rejectedCandidates, cameraMatrix, distanceCoefficients);
        }

        int markerOfBoardDetected = 0;
        if(markerIds.size()>0)
        {
            markerOfBoardDetected = aruco::estimatePoseBoard(markerCorners, markerIds, gridboard, cameraMatrix, distanceCoefficients,
                rvec, tvec);

            //get the quad of the marker board area
            //vector<Point3f> objPoints;
            vector<Point2f> imgPoints;
            //aruco::getBoardObjectAndImagePoints(gridboard, markerCorners, markerIds, objPoints, imgPoints);
            for(int i=0;i<markerCorners.size();i++)
            {
                for(int j=0;j<markerCorners[i].size();j++)
                    imgPoints.push_back(markerCorners[i][j]);
            }

            //using rotate rect
            RotatedRect rect = minAreaRect(imgPoints);

            //cout << rect.size << endl;

            rect.points(outputQuad);

            //add point order to tl, tr, br, rl
            vector<Point2f> beforeOrder = {outputQuad[0], outputQuad[1], outputQuad[2], outputQuad[3]};
            order_points(beforeOrder);
            for(int i=0;i<4;i++)
            {
                outputQuad[i] = beforeOrder[i];
            }

            //cout << logo.size << endl;

            //calculate alpha of the frame and apply alpha to logo
            float ref_density = mean(mean(frame))[0];
            float alpha = ref_density/logo_density;

            //cout << alpha << endl;
            Mat logoAlpha = logo * alpha;

            imshow("logo", logoAlpha);

            //perspective transform
            Mat M = getPerspectiveTransform(inputQuad, outputQuad);
            Mat logoResized;
            warpPerspective(logoAlpha, logoResized, M, frame.size(), INTER_LINEAR, BORDER_REPLICATE);

            imshow("logo", logoResized);

            // Mat mask = Mat::ones(logoAlpha.cols, logoAlpha.rows, CV_8UC3);
            // Mat maskWithBorder;
            // int border = 1;
            // copyMakeBorder(mask, maskWithBorder, border, border, border, border, BORDER_CONSTANT);  

            Mat mask(frame.rows, frame.cols, CV_8UC3, Scalar(0,0,0));
            Scalar color = Scalar(255,255,255);
            Scalar borderColor = Scalar(128, 128, 128);

            vector<Point> points;
            
            for(int i=0;i<4;i++)
            {
                points.push_back((Point)outputQuad[i]);
            }

            //calculate border
            vector<Point> borderPoints;
            int border = 5;
            //LT
            Point lt = Point(outputQuad[0].x - border, outputQuad[0].y - border);
            Point rt = Point(outputQuad[1].x + border, outputQuad[1].y - border);
            Point rb = Point(outputQuad[2].x + border, outputQuad[2].y + border);
            Point lb = Point(outputQuad[3].x - border, outputQuad[3].y + border);
            borderPoints.push_back(lt);
            borderPoints.push_back(rt);
            borderPoints.push_back(rb);
            borderPoints.push_back(lb);

            fillConvexPoly(mask, borderPoints, borderColor);

            putText(mask, "0", lt, FONT_HERSHEY_PLAIN, 1, Scalar(0,255,255));
            putText(mask, "1", rt, FONT_HERSHEY_PLAIN, 1, Scalar(0,255,255));
            putText(mask, "2", rb, FONT_HERSHEY_PLAIN, 1, Scalar(0,255,255));
            putText(mask, "3", lb, FONT_HERSHEY_PLAIN, 1, Scalar(0,255,255));

            //cout << points << endl;

            fillConvexPoly(mask, points, color);

            // Mat maskWithBorder;
            // copyMakeBorder(mask, maskWithBorder, 1, 1, 1, 1, BORDER_CONSTANT, Scalar(255,0,0));

            //imshow("mask", mask);

            Mat mask2(frame.rows, frame.cols, CV_8UC3, Scalar(255,255,255));

            //Mat reverseMask = mask2 - mask;
            Mat reverseMask;// = abs(mask);
            absdiff(mask2, mask, reverseMask);
            // imshow("mask", reverseMask);
            imshow("after", mask);

            // mask = mask / 255;
            // reverseMask = reverseMask / 255;

            //Mat blended = frame * (1-maskWithBorderResized) + logoResized * maskWithBorderResized;

            // alpha blending
            // cout << " Ready to blending " << endl;
            // cout << frame.channels() << endl;
            // cout << reverseMask.channels() << endl;
            // cout << logoResized.channels() << endl;
            // cout << maskWithBorderResized.channels() << endl;

            //Mat blended1, blended2;
            // multiply(frame, mask, blended1);
            //multiply(logoResized, mask, blended2);
            //Mat blended = blended1 + blended2;

            //blended1 = frame.mul(mask);

            Mat blended1(frame.rows, frame.cols, CV_16UC3, Scalar(0,0,0));
            Mat blended2(frame.rows, frame.cols, CV_16UC3, Scalar(0,0,0));

            imshow("mask", reverseMask);

            blended1 = frame.mul(reverseMask) / 255;
            blended2 = logoResized.mul(mask) / 255;

            blended = blended1 + blended2;

            // cout << frame.size << endl;
            // cout << blended.size << endl;


            // for(int r=0;r<frame.rows;r++)
            // {
            //     for(int c=0;c<frame.cols;c++)
            //     {
            //         // blended.at<Vec3b>(r,c) = frame.at<Vec3b>(r,c) * reverseMask.at<Vec3b>(r,c))
            //         //     + logoResized.at<Vec3b>(r,c) * (mask.at<Vec3b>(r,c));

            //     }
            // }


            imshow("after", blended);
        }

        frame.copyTo(frameCopy);

        if(markerIds.size()>0)
        {
            //aruco::drawDetectedMarkers(frameCopy, markerCorners, markerIds);
        }

        if(showRejected && rejectedCandidates.size() > 0)
        {
            aruco::drawDetectedMarkers(frameCopy, rejectedCandidates, noArray(), Scalar(100, 0, 255));
        }

        if(markerOfBoardDetected > 0)
        {
            aruco::drawAxis(frameCopy, cameraMatrix, distanceCoefficients, rvec, tvec, axisLength);
        }

        for(int i=0;i<4;i++)
        {
            line(frameCopy, outputQuad[i], outputQuad[(i+1)%4], Scalar(255, 0, 0), 2);
        }

        imshow("before", frameCopy);
        
        //imshow("after", result);

        if(waitKey(30) >= 0){
            imwrite("sample.jpg", blended);
            break;
        }
    }

    return 0;
    }
    catch(Exception& ex)
    {
        cout << ex.what() << endl;
    }
}
