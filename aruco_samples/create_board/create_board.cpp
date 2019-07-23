#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>

using namespace cv;

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

    if(showImage) {
        imshow("board", boardImage);
        waitKey(0);
    }

    imwrite(out, boardImage);

    return 0;
}
