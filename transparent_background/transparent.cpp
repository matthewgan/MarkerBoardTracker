#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>

using namespace cv;

int main( int argc, char** argv )
{
 char* imageName = argv[1];

 Mat image;
 image = imread( imageName, 1 );

 if( argc != 2 || !image.data )
 {
   printf( " No image data \n " );
   return -1;
 }

 Mat bgr[3];
 split(image, bgr);

//  Mat gray_image;
//  cvtColor( image, gray_image, COLOR_BGR2BGRA );
 

//  imwrite( "./image.png", gray_image );

 imwrite("red.png",bgr[2]); //red channel

 namedWindow( imageName, WINDOW_GUI_NORMAL );
 namedWindow( "Red image", WINDOW_GUI_NORMAL );

 imshow( imageName, image );
 imshow( "Red image", bgr[2] );

 waitKey(0);

 return 0;
}