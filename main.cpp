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

#include "tracker.hpp"
#include "functions.hpp"
#include "registration.hpp"
using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;
// using namespace tracker;

// Static variables
bool Tracker::g_selectObject = false;
int Tracker::g_initTracking = 0;
int Tracker::g_selId = -1;
Rect Tracker::g_selRect;
Point Tracker::g_selOrigin;

// 1. 一个c文件需要调用另一个C文件中的变量或函数,而不能从H文件中调用变量
// 2. 若不想让其他C文件引用本文件中的变量, 加上static即可
// 所以表示下面的这两个变量是定义在其他C文件中的
extern string hot_keys;
extern const char *keys;

int main(int argc, const char **argv)
{

    VideoCapture cap;
    Tracker objTracker;
    // opencv中定义的命令行解析函数
    CommandLineParser parser(argc, argv, keys);
    if (parser.has("help")) {
        help();
        return 0;
    }

    cap.open(argv[1]);
    if (!cap.isOpened()) {
        help();
        cout << "***Could not access file...***\n";
        return -1;
    }
    // 获得视频的宽高
    Size S = Size((int) cap.get(CV_CAP_PROP_FRAME_WIDTH),    //Acquire input size
                  (int) cap.get(CV_CAP_PROP_FRAME_HEIGHT));
    // 输出定义好的热键   
    cout << hot_keys;
    bool paused = false;
    // 从视频中截取图像帧
    Mat frame;
    // 提取第一帧作为后来截取的对象
    cap >> frame;
    // 初始化跟踪器
    // 1. 定义卡尔曼滤波参数
    // 2. 显示控制界面
    // 3. 监听鼠标,调用选择目标回调函数
    objTracker.Init(S, Tracker::InitParams());

    // static_cast用来把 cap.get(CV_CAP_PROP_FOURCC) 转换为int类型
    // 但没有运行时类型检查来保证转换的安全性
    // static_cast可以用来隐式实现任何类型转换
    // cap.get(CV_CAP_PROP_FOURCC) 获得视频的编码格式
    // int ex = static_cast<int>(cap.get(CV_CAP_PROP_FOURCC));
    // ?????? 把输出转换成视频会报错
    VideoWriter outputVideo;
    outputVideo.open("output.mp4" , 0x00000021, cap.get(CV_CAP_PROP_FPS), S, true);

    Mat out;
    try {

        while (1) {
            // 若没有停止且已初始化
            if (!paused && Tracker::g_initTracking) {
                // 提取帧
                cap >> frame;
                if (frame.empty())
                    break;
            }
            // 鼠标选取的过程也会显示出来
            if (!paused) {
                // 处理每一帧
                objTracker.ProcessFrame(frame, out);

            }
            imshow("CamShift", out);
            
            outputVideo << out;

            char c = (char)waitKey(10);
            if (c == 27)
                break;
            switch (c) {
            case 'b':
                // 展示反向投影图
                // 更改标志位
                objTracker.ToggleShowBackproject();
                break;
            case 'c':
                // trackObject = 0;
                // histimg = Scalar::all(0);
                break;
            case 'h':
                // 隐藏控制面板
                objTracker.HideControlsGUI();
                // ----showHist未定义
                // showHist = !showHist;    
                // if (!showHist)
                //     destroyWindow("Histogram");
                // else
                //     namedWindow("Histogram", 1);
                // break;
            // 暂停/重启跟踪 
            case 'p':
                paused = !paused;
                break;
            // 可以重新定义要追踪的物体
            case 'r':
                // 让视频帧从第一帧开始 
                cap.set(CV_CAP_PROP_POS_AVI_RATIO, 0);
                outputVideo.set(CV_CAP_PROP_POS_AVI_RATIO, 0);
                cap >> frame;
                objTracker.Init(S, Tracker::InitParams());

                break;
            default:
                ;
            }
        }
    }

    catch (const cv::Exception &e) {
        std::cerr << e.what();
        cap.release();
        outputVideo.release();

        return 1;
    }
    cap.release();
    outputVideo.release();

    return 0;
}
