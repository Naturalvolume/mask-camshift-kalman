#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utility.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include <iostream>
#include <ctype.h>

// #include "tracker.hpp"
// #include "functions.hpp"
// #include "registration.hpp"
using namespace cv;
using namespace std;
// using namespace cv::xfeatures2d;


int main(int argc, const char **argv)
{
    Mat in_image, search_image, hsv_image, mask, hue, hist, histimg, backproj;
    // Usage： <cmd> <file_in> <file_out>
    //读取原始图像
    in_image = imread(argv[1], IMREAD_UNCHANGED);
    if (in_image.empty()) {
        //检查是否读取图像
        cout << "Error! Input image cannot be read...\n";
        return -1;
    }
    //创建两个具有图像名称的窗口
    // namedWindow("原图", WINDOW_AUTOSIZE);
    // namedWindow("1_hsv", WINDOW_AUTOSIZE);
    // namedWindow("2_mask", WINDOW_AUTOSIZE);
    // namedWindow("3_hue", WINDOW_AUTOSIZE);
    cvtColor(in_image, hsv_image, COLOR_BGR2HSV);
    int _vmin = 32, _vmax = 256;
    // inRange()函数的功能是检查输入数组的每个元素是不是在给定范围内。
    // 检查的是hsv的像素的Hue分量是否在0-180之间，Saturation分量是否在smin-256之间，Value分量是否在MIN(_vmin, _vmax)和MAX(_vmin, _vmax)之间
    // 返回验证矩阵mask，如果hsv的像素点满足条件，那么mask矩阵中对应位置的点置255，不满足条件的置0。
    // 这边的HSV范围是opencv中规定的，因此Hue的范围是0-180，Saturation和Value的范围是0-255。

    inRange(hsv_image, Scalar(0, 50, MIN(_vmin, _vmax)),
        Scalar(180, 256, MAX(_vmin, _vmax)), mask);//确保在范围内
        int ch[] = { 0, 0 };
    hue.create(hsv_image.size(), hsv_image.depth());
    mixChannels(&hsv_image, 1, &hue, 1, ch, 1);//将hsv中的h通道放到hue中去，“提取” h（色调）分量
    // 通道复制函数mixChannels()，此函数由输入参数复制某通道到输出参数特定的通道中
    // 482, 87, 891, 300
    cv::Rect selection = cv::Rect(482, 87, 409, 231);
    rectangle(in_image, selection, Scalar(0, 0, 255), 1, 1, 0);
    Mat roi(hue, selection), maskroi(mask, selection), origin_roi(in_image, selection);

    float histRanges[2];
    histRanges[0] = 0;
    histRanges[1] = 180.0f;
    const float *phranges = histRanges;
    int hSize = 16;
    // 利用calcHist()函数计算了直方图之后再归一化到0-255
    calcHist(&roi, 1, 0, maskroi, hist, 1, &hSize, &phranges);
    normalize(hist, hist, 0, 255, NORM_MINMAX); 

    histimg = Mat::zeros(200, 320, CV_8UC3);
    histimg = Scalar::all(0);
    int binW = histimg.cols / hSize;//一个bin占的宽度
    Mat buf(1, hSize, CV_8UC3);//定义一个缓冲单bin矩阵，1行16列
    for (int i = 0; i < hSize; i++)
        buf.at<Vec3b>(i) = Vec3b(saturate_cast<uchar>(i*180. / hSize), 255, 255);
    cvtColor(buf, buf, COLOR_HSV2BGR);

    for (int i = 0; i < hSize; i++)
    {
        int val = saturate_cast<int>(hist.at<float>(i)*histimg.rows / 255);//获取直方图相对高度
        rectangle(histimg, Point(i*binW, histimg.rows),
        Point((i + 1)*binW, histimg.rows - val),//画出直方图，左上角坐标，右下角坐标，高度，颜色，大小，线型
        Scalar(buf.at<Vec3b>(i)), -1, 8);
    }
    // 原图的反向投影
    // calcBackProject(&hue, 1, 0, hist, backproj, &phranges);

    // 读入第二帧
    search_image = imread(argv[2], IMREAD_UNCHANGED);
    cvtColor(search_image, hsv_image, COLOR_BGR2HSV);
    inRange(hsv_image, Scalar(0, 50, MIN(_vmin, _vmax)),
        Scalar(180, 256, MAX(_vmin, _vmax)), mask);//确保在范围内
    hue.create(hsv_image.size(), hsv_image.depth());
    mixChannels(&hsv_image, 1, &hue, 1, ch, 1);
    calcBackProject(&hue, 1, 0, hist, backproj, &phranges);


    //在之前创建的窗口中显示图片
    // imshow("原图", in_image);
    // waitKey(); // Wait for key press
    //写入图像
    cout << "ready" << endl;
    // imwrite("/home/kathy/happy/dataset/red_car/camshift/8_hsv.png", hsv_image);
    // imwrite("/home/kathy/happy/dataset/red_car/camshift/9_mask.png", mask);
    // imwrite("/home/kathy/happy/dataset/red_car/camshift/10_hue.png", hue);
    // imwrite("/home/kathy/happy/dataset/red_car/camshift/4_roi.png", in_image);
    // imwrite("/home/kathy/happy/dataset/red_car/camshift/5_roi.png", origin_roi);
    // imwrite("/home/kathy/happy/dataset/red_car/camshift/6_hist.png", histimg);
    imwrite("/home/kathy/happy/dataset/red_car/camshift/12_backproj.png", backproj);
    return 0;

}
