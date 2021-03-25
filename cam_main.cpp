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
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
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

// 1.15
void LoadImages(const string &strSequence, vector<string> &vstrImageFilenames, const string &strSemantic, vector<string> &vstrSemanticFile,
        vector<string> &vstrLoaction, const string &count);
void LoadMask(const string &strFilenamesMask, cv::Mat &imMask);

int main(int argc, const char **argv)
{

    Tracker objTracker;
    // opencv中定义的命令行解析函数
    CommandLineParser parser(argc, argv, keys);
    if (parser.has("help") || argc != 4) {
        help();
        return 0;
    }
    // 输出定义好的热键   
    cout << hot_keys;
    bool paused = false;

    vector<string> vstrImageFilenames;
    vector<string> vstrSemanticFile;
    vector<string> vstrLocation;
    LoadImages(string(argv[1]), vstrImageFilenames, string(argv[2]), vstrSemanticFile, vstrLocation, string(argv[3]));

    int nImages = vstrImageFilenames.size();
    // 验证图片和语义数量是否正确
    if(vstrImageFilenames.empty())
    {
        cerr << endl << "error1: No images found in provided path." << endl;
        return 1;
    }
    else if(vstrSemanticFile.empty())
    {
        cerr << endl << "error2: No semanticFile found in provided path." << endl;
        return 1;
    }
    else if(vstrImageFilenames.size() != vstrSemanticFile.size() || vstrLocation.size() != nImages)
    {
        cerr << endl << "error3: Different number of images for segmentation." << endl;
        return 1;
    }

    // TODO 输出
    // VideoWriter outputVideo;
    // outputVideo.open("output.mp4" , 0x00000021, cap.get(CV_CAP_PROP_FPS), S, true);

    Mat frame, out;
    for(int ni=0; ni<nImages; ni++) {
        // read image
        frame = imread(vstrImageFilenames[ni], CV_LOAD_IMAGE_UNCHANGED);
        // cout << "clos:" << frame.cols << endl;
	    // cout << "rows" << frame.rows << endl;
        cv::namedWindow("CamShift", cv::WINDOW_AUTOSIZE);
        if(frame.empty()) {
            cerr << endl << "Failed to load image at: " << vstrImageFilenames[ni] << endl;
            return 1;
        }

        ifstream file_location;
        file_location.open(vstrLocation[ni].c_str());
        while(!file_location) {
            cout << "the file is not open" << endl;
        }
        Point p1, p2;
        while(!file_location.eof()) {
            string s;
            getline(file_location, s);
            if(!s.empty()) {
                vector<string> vStr;
                boost::split(vStr, s, boost::is_any_of(" "), boost::token_compress_on);
                p1.x = atoi(vStr[2].c_str());
                p1.y = atoi(vStr[3].c_str());
                p2.x = atoi(vStr[4].c_str());
                p2.y = atoi(vStr[5].c_str());
            }
            // TODO 这是只能处理一个跟踪物体的

        }
        
        Tracker::g_selRect = cv::Rect(p1, p2);
        // ----这是显示跟踪物体矩形,仅供调试用
        rectangle(frame, Tracker::g_selRect, Scalar(0, 0, 255), 1, 1, 0);
        
        // cv::namedWindow("roi", cv::WINDOW_AUTOSIZE);
        // imshow("roi", frame);
        // waitKey(2000);
        // 没有暂停跟踪且未初始化目标
        if (!paused && !Tracker::g_initTracking) {
            cv::Size S = cv::Size(frame.rows, frame.cols);
            objTracker.Init(S, Tracker::InitParams());
        }
        if (!paused) {
            // 处理每一帧
            cout << "processing:" << ni << endl;
            objTracker.ProcessFrame(frame, out);
            waitKey(2000);

        }
        imshow("CamShift", out);
        
            
        // outputVideo << out;
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
            // cap.set(CV_CAP_PROP_POS_AVI_RATIO, 0);
            // outputVideo.set(CV_CAP_PROP_POS_AVI_RATIO, 0);
            // cap >> frame;
            // objTracker.Init(S, Tracker::InitParams());
            break;
        default:
            ;
        }
    }

  
    // 初始化跟踪器
    // 1. 定义卡尔曼滤波参数
    // 2. 显示控制界面
    // 3. 监听鼠标,调用选择目标回调函数
    
    

    // static_cast用来把 cap.get(CV_CAP_PROP_FOURCC) 转换为int类型
    // 但没有运行时类型检查来保证转换的安全性
    // static_cast可以用来隐式实现任何类型转换
    // cap.get(CV_CAP_PROP_FOURCC) 获得视频的编码格式
    // int ex = static_cast<int>(cap.get(CV_CAP_PROP_FOURCC));
    // ?????? 把输出转换成视频会报错
    

    // outputVideo.release();
    return 0;
}

void LoadImages(const string &strSequence, vector<string> &vstrImageFilenames, const string &strSemantic, vector<string> &vstrSemanticFile,
        vector<string> &vstrLoaction, const string &count)
{
    // TODO 时间戳文件怎么弄
    

    // 输入的路径要到对应的图片文件夹
    string strPrefixLeft = strSequence;
    string strPrefixSemantic = strSemantic;

    int c = atoi(count.c_str());
    vstrImageFilenames.resize(c);
    vstrSemanticFile.resize(c);
    vstrLoaction.resize(c);

    for(int i=0; i<c; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(6) << i;
        vstrImageFilenames[i] = strPrefixLeft + ss.str() + ".png";
        // cout << vstrImageFilenames[i] << endl;
        vstrSemanticFile[i] = strPrefixSemantic + ss.str() + ".txt";
        // cout << vstrSemanticFile[i] << endl;
        vstrLoaction[i] = strPrefixSemantic + ss.str() + "l.txt";
        cout << vstrLoaction[i] << endl;
    }
}

void LoadMask(const string &strFilenamesMask, cv::Mat &im)
{
    ifstream file_mask;
    file_mask.open(strFilenamesMask.c_str());

    // Main loop
    // count代表图片的行数
    int count = 0;
    // ｉｍｇＬａｂｅｌ是为了展示
    // cv::Mat imgLabel(im.rows,im.cols,CV_8UC3); // for display
    while(!file_mask.eof())
    {
        string s;
        getline(file_mask, s);
        
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            int tmp;
            // 根据掩码图片的矩阵列宽，遍历读取每个变量
            for(int i = 0; i < im.cols; ++i){
                ss >> tmp;
                if (tmp!=0){
                    im.at<uchar>(count,i) = 0;
                   
                }
            }
            count++;
        }
    }
    return;
}