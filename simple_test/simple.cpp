#include <iostream>
#include <opencv2/aruco.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv){
    
    VideoCapture inputVideo;
    inputVideo.open(2);

    Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::PREDEFINED_DICTIONARY_NAME::DICT_6X6_250);
    while (inputVideo.grab())
    {
        /* code */
        cv::Mat image, imageCopy;
        inputVideo.retrieve(image);
        image.copyTo(imageCopy);

        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners;
        
        cv::aruco::detectMarkers(
            image, 
            dictionary,
            corners, 
            ids);

        cout << ids.size() << endl;

        if(ids.size()>0){
            cv::aruco::drawDetectedMarkers(imageCopy, corners, ids);
        }

        cout << ids.size() << endl;

        cv::imshow("out", imageCopy);

        char key = cv::waitKey(1);

        if(key==27) break;
    }
}