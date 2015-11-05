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


using namespace cv;
using namespace std;


struct Pixel{
    float dX,dY;
    Mat sumTensor;
    float R;
    bool inImg;
};

vector<Pixel> pixValues;

cv::Mat inputImage, workImage, outputImage;
vector<float> GaussMask;

int sigma_value, threshold_value, alpha_value;
const double sigma_max = 1000, alpha_max=20, threshold_max=100;
char thresholdTrackbar[20]="treshold %", sigmaTrackbar[20] = "sigma 1/1000", alphaTrackbar[20]="alpha 0.4-0.6";
int windowWidth;

void changeSigma(int sigma, void*);
void changeAlpha(int alpha, void*);
void changeThreshold(int threshold, void*);



Vec3f getColor(int y,int x){
    if (x<0 || x>= workImage.cols || y<0 || y>= workImage.rows){
        return Vec3f(0,0,0);
    }
    return workImage.at<Vec3f>(y, x) / 255.0f;
}

//calc windowwidth out of sigma slider settings
void calcWindowWidth(){
    windowWidth= (int) 1+2*ceil(sigma_value*sqrt(-2*log(0.1)));
}

//calc weight for each window pixel
void calcGaussMask(){
    int delta=(windowWidth-1)/2;
    int x=ceil(windowWidth/2);
    int y=ceil(windowWidth/2);
    int cx,cy;
    for(int i=0; i<windowWidth*windowWidth; i++){ //i for window
        cy=y-delta+floor(i/windowWidth);
        cx=x-delta+(i % windowWidth);
        GaussMask.push_back(exp(-(pow(cx,2)+pow(cy,2))/2*pow(sigma_value,2))/2*M_PI*pow(sigma_value,2));
    }
}

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
            for(int i=0; i<workImage.channels(); i++){
                dy = max((bottom.val[i]-top.val[i])/2, dy);
                dx = max((right.val[i]-left.val[i])/2, dx);
            } //TODO test different settings like average, addition. etc
        pixValues.at(y*workImage.cols+x).dX = dx;
        pixValues.at(y*workImage.cols+x).dY = dy;
        }
    }
}

void calcTensor(){
    //calculate Tensor
    int cy,cx, delta=(windowWidth-1)/2;
    Mat tensor;
    Mat tmpMat= Mat(1,3,CV_32FC1, Scalar(0));
    Pixel pix;

    for(int y=0; y < workImage.rows; y++){
        for(int x=0; x<workImage.cols; x++){
            tensor = Mat(1,3,CV_32FC1, Scalar(0));

            for(int i=0; i<windowWidth*windowWidth; i++){ //i for window
                cy=y-delta+floor(i/windowWidth);
                cx=x-delta+(i % windowWidth);

                if(cx<0 || cy<0 || cx>=workImage.cols || cy>=workImage.rows){continue;}

                pix = pixValues.at(cy*workImage.cols+cx);
                tmpMat.at<float>(0,0) =  pix.dX * pix.dX * GaussMask.at(i);
                tmpMat.at<float>(0,1)= pix.dX * pix.dY * GaussMask.at(i);
                tmpMat.at<float>(0,2) = pix.dY * pix.dY * GaussMask.at(i);

                tensor += tmpMat ;
            }
            pixValues.at(y*workImage.cols+x).sumTensor = tensor;
        }
    }
}


void changeSigma(int sigmaSliderValue, void * ){
    sigma_value = sigmaSliderValue*(4-0.1)/sigma_max + 0.1;
    calcWindowWidth();
    calcGaussMask();
    calcDerivatives();
    calcTensor();
    changeAlpha(alpha_value,0);
}

void calcScore(int alpha){
    float det, sqTrace;
    for(int y=0; y < workImage.rows; y++){
        for(int x=0; x<workImage.cols; x++){
            Pixel pix = pixValues.at(y*workImage.cols+x);
            det= pix.sumTensor.at<float>(0,0) * pix.sumTensor.at<float>(0,2) - pix.sumTensor.at<float>(0,1)*pix.sumTensor.at<float>(0,1);
            sqTrace = (pix.sumTensor.at<float>(0,0)+pix.sumTensor.at<float>(0,2))*(pix.sumTensor.at<float>(0,0)+pix.sumTensor.at<float>(0,2)); //trace*trace
              pixValues.at(y*workImage.cols+x).R = det - alpha*sqTrace;
        }
    }
}

void changeAlpha(int alphaSliderValue, void * ){
    alpha_value=alphaSliderValue/200 ;
    calcScore(alpha_value);
    changeThreshold(threshold_value,0);
}

bool higherThanNeighbour(int y, int x){
    int nb[8][2]= {{y-1,x-1},{y-1,x},{y-1,x+1},{y,x-1},{y,x+1},{y+1,x-1},{y+1,x},{y+1,x+1}};

    for(int i=0; i<8; i++){
        if(nb[i][0]<0 || nb[i][1]<0 || nb[i][0]>=workImage.rows || nb[i][1]>=workImage.cols ){continue;}
        if(pixValues.at(nb[i][0]*workImage.cols+nb[i][1]).R > pixValues.at(y*workImage.cols+x).R){return false;}
    }
    return true;

}

void changeThreshold(int threshold, void *){

    for(int y=0; y < workImage.rows; y++){
        for(int x=0; x<workImage.cols; x++){
            if(true){
             outputImage.at<Vec3f>(y,x)= getColor(y,x);
// higherThanNeighbour(y,x) && pixValues.at(y*workImage.cols+x).R>threshold
            }
        }
    }
    imshow( "Harris Corner Detector", workImage );
}


int main( int argc, char** argv ){
    string imageName("./fruits.jpg"); // by default
    if( argc > 1){
        imageName = argv[1];
    }
    inputImage = imread(imageName.c_str(), IMREAD_COLOR);
    inputImage.convertTo(workImage, CV_32FC3);

    namedWindow("test", WINDOW_AUTOSIZE );

    imshow("test", workImage);
    std::cout<<"test"<<std::endl;
    outputImage = Mat(workImage.rows, workImage.cols,CV_32FC3, Scalar(0,0,0));
    if( inputImage.empty() ){
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }


    //build Window
    namedWindow("Harris Corner Detector", WINDOW_AUTOSIZE );
    //build Trackbars
    sigma_value=100;
    alpha_value = 1;
    threshold_value =1;
    createTrackbar(sigmaTrackbar, "Harris Corner Detector", &sigma_value , sigma_max, changeSigma );
    createTrackbar(alphaTrackbar, "Harris Corner Detector",&alpha_value, alpha_max, changeAlpha );
    createTrackbar(thresholdTrackbar, "Harris Corner Detector", &threshold_value, threshold_max , changeThreshold );
    //build Image functions here!
   changeSigma(sigma_value ,0);
    waitKey(0); // Wait for a keystroke in the window
    return 0;
}
