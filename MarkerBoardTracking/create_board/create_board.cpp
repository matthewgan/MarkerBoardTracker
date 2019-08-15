#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char *argv[]) {
    
    int markersX = 4;
    int markersY = 6;
    int markerLength =160; //pixels
    int markerSeparation = 25; //pixels
    int dictionaryId = aruco::PREDEFINED_DICTIONARY_NAME::DICT_6X6_50;
    int margins = markerSeparation;

    int borderBits = 1;
    bool showImage = true;

    String out = "board.jpg";

    Size imageSize;
    imageSize.width = markersX * (markerLength + markerSeparation) - markerSeparation + 2 * margins;
    imageSize.height = markersY * (markerLength + markerSeparation) - markerSeparation + 2 * margins;

    Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));

    Ptr<aruco::GridBoard> board = aruco::GridBoard::create(markersX, markersY, float(markerLength),
                                                      float(markerSeparation), dictionary);

    // show created board
    Mat boardImage;
    board->draw(imageSize, boardImage, margins, borderBits);

    cout << boardImage.channels() << endl;

    vector<Mat> channels;

    channels.push_back(boardImage);
    channels.push_back(boardImage);
    channels.push_back(boardImage);

    Mat boardGreenImage;
    merge(channels, boardGreenImage);

    cout << boardGreenImage.channels() << endl;
    cout << boardGreenImage.rows << endl;
    cout << boardGreenImage.cols << endl;
    //cout << boardGreenImage << endl;

    for(int r=0;r<boardGreenImage.rows;r++)
    {
        for(int c=0;c<boardGreenImage.cols;c++)
        {
            Vec3b color = boardGreenImage.at<Vec3b>(Point(c,r));
            if(color[0] > 150 && color[1] > 150 && color[2] > 150)
            {
                color[0] = 0;
                color[1] = 255;
                color[2] = 0;
                //cout << "Pixel >200 :" << r << "," << c << endl;
            }
            else
            {
                color.val[0] = 255;
                color.val[1] = 255;
                color.val[2] = 0;
            }
            boardGreenImage.at<Vec3b>(Point(c,r)) = color;
        }
    }

    if(showImage) {
        imshow("board", boardGreenImage);
        waitKey(0);
    }

    imwrite(out, boardGreenImage);

    return 0;
}
