#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/utility.hpp>
#include <queue>
#include <iostream>
#include <string>
#include <list>
#include <vector>
#include <stdio.h>


/*
algo: Harris Corner Detector
author: Josephine Rehak

*/

using namespace cv;
using namespace std;

//holds data structure for each pixel in image
struct Pixel{
    float dX,dY;
    Mat sumTensor;
    float R;
};

int channels = 3.0;

vector<Pixel> pixValues; //holds set of pixels of image

cv::Mat inputImage, workImage, outputImage; // working matrices
vector<float> GaussMask; // holds values of GausMask

int sigma_value, threshold_value, alpha_value; // holds slider values
const double sigma_max = 100, alpha_max=20, threshold_max=30; // max slider values
char thresholdTrackbar[25]="treshold:5 to 5^-25", sigmaTrackbar[20] = "sigma: 1/1000", alphaTrackbar[20]="alpha: 0.4-0.6";
int windowWidth; // width of quadtratic window

void changeSigma(int sigma, void*);
void changeAlpha(int alpha, void*);
void changeThreshold(int threshold, void*);

Vec3f getColor(int y,int x){ // get color of Pixel in original Image on x,y
    if (x<0){x=0;}
    if (x>= workImage.cols){x=workImage.cols;}
    if (y<0){x=0;}
    if (y>= workImage.rows){y=workImage.rows;}
    return workImage.at<Vec3f>(y, x);
}

//calc windowwidth out of sigma slider settings
void calcWindowWidth(){
    windowWidth= (int) 1+2*ceil(sigma_value*sqrt(-2*log(0.2)));
}


//calc weight for each window pixel + build Gaussmask
void calcGaussMask(){
    int delta=(windowWidth-1)/2;
    int x=ceil(windowWidth/2);
    int y=ceil(windowWidth/2);
    int cx,cy;
    for(int i=0; i<windowWidth*windowWidth; i++){ //i for window
        cy=y-delta+floor(i/windowWidth);
        cx=x-delta+(i % windowWidth);
        float calc = exp(-(pow(cx,2)+pow(cy,2))/2*sigma_value*sigma_value)/2.0*M_PI*sigma_value*sigma_value;
        GaussMask.push_back(calc);
    }
}

// calculates dx + dy out of gradient and colors for each pixel
void calcDerivatives(){
    Vec3f top, bottom, left, right;
    for(int y=0; y < workImage.rows; y++){
        for(int x=0; x<workImage.cols; x++){
            pixValues.push_back(Pixel());

            //calculate Derivatives
            top = getColor(y-1,x);
            left = getColor(y,x-1);
            right = getColor(y,x+1);
            bottom = getColor(y+1,x);
            float dx = 0;
            float dy = 0;
            for(int i=0; i<channels; i++){
                dy +=(bottom.val[i]-top.val[i])/2;
                dx += (right.val[i]-left.val[i])/2;
            } //TODO test different settings like average, addition. etc
        pixValues.at(y*workImage.cols+x).dX = dx/channels;
        pixValues.at(y*workImage.cols+x).dY = dy/channels;

        }
    }
}

// calculates Tensor for each pixel
void calcTensor(){
    int cy,cx, delta=(windowWidth-1)/2;
    Mat tensor;
    Mat tmpMat= Mat(1,3,CV_32FC1, Scalar(0)); // values for dx*dx dx*dy... in 3x1 matrix
    Pixel pix;

    for(int y=0; y < workImage.rows; y++){
        for(int x=0; x<workImage.cols; x++){
            tensor = Mat(1,3,CV_32FC1, Scalar(0));
            // go through all pixels in window
            for(int i=0; i<windowWidth*windowWidth; i++){ //i for window
                cy=y-delta+floor(i/windowWidth);
                cx=x-delta+(i % windowWidth);

                if(cx<0 || cy<0 || cx>=workImage.cols || cy>=workImage.rows){continue;}

                pix = pixValues.at(cy*workImage.cols+cx);

                tmpMat.at<float>(0,0) = pix.dX * pix.dX * GaussMask.at(i);
                tmpMat.at<float>(0,1) = pix.dX * pix.dY * GaussMask.at(i);
                tmpMat.at<float>(0,2) = pix.dY * pix.dY * GaussMask.at(i);

                tensor += tmpMat ;
            }
            pixValues.at(y*workImage.cols+x).sumTensor = tensor;


        }
    }
}

// function called when sigma on slider changed
void changeSigma(int sigmaSliderValue, void * ){
    sigma_value = sigmaSliderValue*(4-0.1)/sigma_max + 0.1;
    calcWindowWidth();
    calcGaussMask();
    calcDerivatives();
    calcTensor();
    changeAlpha(alpha_value,0);
}

// calculates score R for each pixel R= det(a) - alpha*trace(A)*trace(A)
void calcScore(int alpha){
    float det, trace;
    for(int y=0; y < workImage.rows; y++){
        for(int x=0; x<workImage.cols; x++){
            Pixel pix = pixValues.at(y*workImage.cols+x);
            det= pix.sumTensor.at<float>(0,0) * pix.sumTensor.at<float>(0,2) - pix.sumTensor.at<float>(0,1)*pix.sumTensor.at<float>(0,1);
            trace = pix.sumTensor.at<float>(0,0)+pix.sumTensor.at<float>(0,2);
              pixValues.at(y*workImage.cols+x).R = det - alpha*trace*trace;
        }
    }
}

// function called when alpha on slider changed
void changeAlpha(int alphaSliderValue, void * ){
    alpha_value=alphaSliderValue/200 +0.4 ;
    calcScore(alpha_value);
    changeThreshold(threshold_value,0);
}

// check score of all neighbours to find maxima
bool higherThanNeighbour(int y, int x){
    int nb[8][2]= {{y-1,x-1},{y-1,x},{y-1,x+1},{y,x-1},{y,x+1},{y+1,x-1},{y+1,x},{y+1,x+1}};

    for(int i=0; i<8; i++){
        if(nb[i][0]<0 || nb[i][1]<0 || nb[i][0]>=workImage.rows || nb[i][1]>=workImage.cols ){continue;}
        if(pixValues.at(nb[i][0]*workImage.cols+nb[i][1]).R > pixValues.at(y*workImage.cols+x).R){return false;}
    }
    return true;
}

// function called when threshold on slider changed
void changeThreshold(int thresholdSliderValue, void *){
    float threshold = pow(5,-(thresholdSliderValue+1));

    for(int y=0; y < workImage.rows; y++){
        for(int x=0; x<workImage.cols; x++){
            if(higherThanNeighbour(y,x) && pixValues.at(y*workImage.cols+x).R>threshold){
              outputImage.at<Vec3f>(y*workImage.cols+x) = Vec3f(0,0,200); //set pixel red
            }else{
                outputImage.at<Vec3f>(y*workImage.cols+x) = Vec3f(0,0,0); // set pixel black
            }
        }
    }
 imshow( "Harris Corner Detector", outputImage );
}


int main( int argc, char** argv ){
    string imageName("./test3.png"); // by default
    if( argc > 1){ imageName = argv[1];}
    inputImage = imread(imageName.c_str(), IMREAD_COLOR);
    if( inputImage.empty() ){
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }
    //build matrices
    inputImage.convertTo(workImage, CV_32FC3, 1/255.0);
    outputImage = Mat(workImage.rows, workImage.cols,CV_32FC3, Scalar(0,0,0));
    //output of original Image
    namedWindow("original", WINDOW_AUTOSIZE);
    imshow("original", workImage);

    //build Window
    namedWindow("Harris Corner Detector", WINDOW_AUTOSIZE );
    //build Trackbars
    sigma_value=40;
    alpha_value = 1;
    threshold_value =6;
    createTrackbar(sigmaTrackbar, "Harris Corner Detector", &sigma_value , sigma_max, changeSigma );
    createTrackbar(alphaTrackbar, "Harris Corner Detector",&alpha_value, alpha_max, changeAlpha );
    createTrackbar(thresholdTrackbar, "Harris Corner Detector", &threshold_value, threshold_max , changeThreshold );
    //first onchange function here!
   changeSigma(sigma_value ,0);
    waitKey(0); // Wait for a keystroke in the window
    return 0;
}
