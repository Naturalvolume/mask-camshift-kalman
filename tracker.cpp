#include "tracker.hpp"
using namespace std;
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utility.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>

// 画线
// 画卡尔曼滤波预测的位置,两条线的交点
#define drawCross( center, color, d )                                 \
line( out, Point( center.x - d, center.y - d ), Point( center.x + d, center.y + d ), color, 2, CV_AA, 0); \
line( out, Point( center.x + d, center.y - d ), Point( center.x - d, center.y + d ), color, 2, CV_AA, 0 )

// 定义Tracker 参数默认值
Tracker::Tracker():
  m_showControlsGUI(false),
  m_initialized(false)
{}

Tracker::InitParams::InitParams()
  : histDims(16),
    vMin(32),
    vMax(256),
    sMin(50),
    sBox(8),
    showBackproject(true),
    showControlsGUI(true),
    showHistogram(true),
    histRanges()
{
  histRanges[0] = 0;
  histRanges[1] = 180.0f;
}
// 初始化
void Tracker::Init(const cv::Size &frameSize,
                   const InitParams &initParams)
{
  // Get frame size
  m_frameSize = frameSize;

  // Load init params.
  m_hSize = 16;
  m_histDims = initParams.histDims;
  m_vMin = initParams.vMin;
  m_vMax = initParams.vMax;
  m_sMin = initParams.sMin;
  m_sBox = initParams.sBox;
  m_histRanges[0] = initParams.histRanges[0];
  m_histRanges[1] = initParams.histRanges[1];

  // 卡尔曼滤波参数
  m_KF = cv::KalmanFilter(4, 2, 0);
  m_state = cv::Mat(2, 1, CV_32F); /* (phi, delta_phi) */
  m_measured = cv::Mat::zeros(2, 1, CV_32F);

  m_KF.transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 0, 1, 0,   0, 1, 0, 1,  0, 0, 1, 0,  0, 0, 0, 1);
  // cv::setIdentity 将行数和列数相等的元素设置为1,即设为单位矩阵
  cv::setIdentity(m_KF.measurementMatrix);
  cv::setIdentity(m_KF.processNoiseCov, cv::Scalar::all(1e-2));
  cv::setIdentity(m_KF.measurementNoiseCov, cv::Scalar::all(10));
  cv::setIdentity(m_KF.errorCovPost, cv::Scalar::all(1));

  // 清除上一帧的卡尔曼滤波参数
  m_past.clear();
  m_kalmanv.clear();

  m_histImg = Mat::zeros(200, 320, CV_8UC3);

  // Show controlsGUI, m_imgBackproject, histogram
  m_showControlsGUI = false;
  // 显示控制面板,选择追踪物体,调用鼠标选择后的回调函数
  ShowControlsGUI();
  // Set initialized flag
  m_initialized = true;

  // Static variables
  // g_selectObject = false;
  g_initTracking = false;
  // g_selId = -1;
  // g_selRect;
  // g_selOrigin;
  // bool Tracker::g_selectObject = false;
  // int Tracker::g_initTracking = 0;
  // int Tracker::g_selId = -1;
  // cv::Rect Tracker::g_selRect;
  // cv::Point Tracker::g_selOrigin;


}

void Tracker::ShowControlsGUI()
{
  // cvNamedWindow(m_controlsGUIWndName.c_str(), 1);
  cv::namedWindow("CamShift", 0);
  // ???? 这里虽然有直方图窗口但是并没有显示直方图,而是黑色图片
  // 这两个窗口并没有使用
  cv::namedWindow("hsv", 0);
  cv::namedWindow("hue", 0);
  cv::namedWindow("Histogram", 0);
  cv::namedWindow("Trackbars", 0);
  if (!m_showControlsGUI) {

    cv::createTrackbar("Vmin", "Trackbars", &m_vMin, 256, 0);
    cv::createTrackbar("Vmax", "Trackbars", &m_vMax, 256, 0);
    cv::createTrackbar("Smin", "Trackbars", &m_sMin, 256, 0);
    // 鼠标选择 回调函数,鼠标选择之后调用onMouse函数
    // ----在这里处理实例分割要追踪的目标
    cv::setMouseCallback("CamShift", Tracker::OnMouse, 0);
  }
  m_showControlsGUI = true;

}
// 析构函数
Tracker::~Tracker()
{
  // Safe free buffers.
  // Deinit();
  // Destroy windows.
  cv::destroyAllWindows();
}
// ???? 这里为啥int后面可以是一个被注释掉的变量
// 保存选择目标的矩形区域,改变目前的跟踪状态
void Tracker::OnMouse(int event, int x, int y, int /*flags*/, void *param)
{
  // 鼠标的监听事件包括按下 抬起
  // 所以会触发两次
  // 第一次触发确定起始点(), 第二次触发g_selectObject为TRUE,根据大小 宽高确定目标物体矩形

  // 第二次:鼠标抬起时
  // 根据当前点x和已选择起点x间的关系确定矩形的终点
  if (g_selectObject) {
    // g_selOrigin是已经选择的物体矩形区域
    g_selRect.x = MIN(x, g_selOrigin.x);
    g_selRect.y = MIN(y, g_selOrigin.y);
    // 用绝对值确定矩形的宽高
    g_selRect.width = std::abs(x - g_selOrigin.x);
    g_selRect.height = std::abs(y - g_selOrigin.y);

  }

  switch (event) {
    // 第一次:监听到鼠标按下事件
  case EVENT_LBUTTONDOWN:
    // 矩形起点 
    g_selOrigin = cv::Point(x, y);
    g_selRect = cv::Rect(x, y, 0, 0);
    // 设置选择物体标志为TRUE,可以在下一次触发事件中计算得到矩形位置和宽高
    g_selectObject = true;
    break;
    // 第二次:鼠标抬起,物体选择结束
  case EVENT_LBUTTONUP:
    g_selectObject = false;
    // 物体选择成功,改变g_initTracking标志位
    // 表示此时已选择好物体,但还未开始跟踪
    if (g_selRect.width > 0 && g_selRect.height > 0)
      g_initTracking = -1;
    break;
  }
}

void Tracker::InitTrackWindow(const cv::Mat &img, const cv::Rect &selRect)
{
  // if (g_initTracking < 0) {
  // roi是原图的h分量
  // maskroi是二值化之后的图像掩码
  cv::Mat roi(m_imgHue, g_selRect), maskroi(m_imgMask, g_selRect);

  // Create histograms
  // 直方图范围
  const float *phranges = m_histRanges;
  //  calcHist(const Mat* images, int nimages, const int* channels, InputArray mask, OutputArray hist, 
  // int dims, const int* histSize, const float** ranges, bool uniform=true, bool accumulate=false )
  // roi:输入图像
  // 1 输入图像个数
  // 0 需要统计直方图的第几通道
  // maskroi:掩码,只计算掩码内的直方图
  // m_hist:输出的直方图数组
  // 1 直方图通道的个数(维度)
  // m_hSize:直方图分成的区间数
  // phranges:统计像素值的区间
  // 计算要跟踪目标的直方图,m_hist就是跟踪目标的直方图
  cv::calcHist(&roi, 1, 0, maskroi, m_hist, 1, &m_hSize, &phranges);
  // 直方图归一化到0-255
  cv::normalize(m_hist, m_hist, 0, 255, NORM_MINMAX);

  // 让跟踪框等于选择目标框
  m_trackWindow = g_selRect;
  // 把跟踪标志位置1,表示目标物体直方图已得到,可以正式开始跟踪了
  g_initTracking = 1;

  //Init state
  // 卡尔曼滤波参数,初始状态为选择目标框的一半
  m_state.at<float>(0) = g_selRect.x + g_selRect.width / 2;
  m_state.at<float>(1) = g_selRect.y + g_selRect.height / 2;

  // -----这里是可视化直方图代码,实际跟踪中并不需要
  m_histImg = Scalar::all(0);
  // 计算得到直方图每次要统计的图像列数
  int binW = m_histImg.cols / m_hSize;
  Mat buf(1, m_hSize, CV_8UC3);
  // ???? 给buf赋值,然后转换成bgr形式,是什么意思呢
  for (int i = 0; i < m_hSize; i++)
    buf.at<Vec3b>(i) = Vec3b(saturate_cast<uchar>(i * 180. / m_hSize), 255, 255);
  cvtColor(buf, buf, COLOR_HSV2BGR);

  // ???根据直方图计算,显示直方图
  for (int i = 0; i < m_hSize; i++) {
    int val = saturate_cast<int>(m_hist.at<float>(i) * m_histImg.rows / 255);// 获取直方图相对高度
    rectangle(m_histImg, Point(i * binW, m_histImg.rows),
              Point((i + 1)*binW, m_histImg.rows - val),//画出直方图,左上角坐标,右下角坐标,高度,颜色,大小,线型
              Scalar(buf.at<Vec3b>(i)), -1, 8);
  }

  imshow("Histogram", m_histImg);
  // }

}

void Tracker::ProcessFrame(const cv::Mat &img, cv::Mat &out)
{

  img.copyTo(out);



  // Draw selection box
  // 若已选择物体
  // ??? 在鼠标监听的时候,当物体选择完毕该标志位是会被置为false的
  // 所以这里啥时候会被调用呢
  // ??? 这里应该不会被调用
  if (g_selectObject) {
    // 且物体符合条件
    if ((g_selRect.width > 0) && (g_selRect.height > 0)) {
      // 截取目标物体为感兴趣区域
      Mat roi(out, g_selRect);
      // bitwise_not表示按位操作像素,对二进制数据进行非操作
      // 对图像(灰度图或彩色图均可)每个像素值进行二进制非操作 ~1=0 ~0=1
      // 即对感兴趣区域的目标图像全部按位取反,即黑的变成白的,白的变成黑的
      // ????? 但是这里
      bitwise_not(roi, roi);
      // 在所有图像基本运算的操作函数中,凡是带有掩码的处理函数,其掩码都参与运算(输入图像运算完之后再与掩码图像或矩阵计算)
    }
  }



  int ch[] = {0, 0};
  // m_imgHSV是视频的HSV格式图像
  cv::cvtColor(img, m_imgHSV, COLOR_BGR2HSV);
  // 二值化HSV,把没看组条件的置255,不满足的置0
  // 输出图像掩码m_imgMask
  // 检查图像的每个像素是否在给定范围内
  // hsv的hue分量是否在0-180之间,saturation分量是否在 m_sMin-256之间,value分量是否在MIN(m_vMin, m_vMax) 和 MAX(m_vMin, m_vMax)之间
  // 返回验证掩码m_imgMask
  cv::inRange(m_imgHSV, cv::Scalar(0, m_sMin, MIN(m_vMin, m_vMax)),
              cv::Scalar(180, 256, MAX(m_vMin, m_vMax)), m_imgMask);
  // create: 分配新的数组数据; 或创建一个图像矩阵的矩阵体 
  // 创建一个指定大小, 指定类型的图像矩阵的矩阵体
  // depth代表每个像素中每个通道的精度,但它本身和图像的通道数无关,depth数值越大,精度越高
  // Mat.depth 得到的是0~6的数字,分别代表不同的位数
  // enum{CV_8U=0,CV_8S=1,CV_16U=2,CV_16S=3,CV_32S=4,CV_32F=5,CV_64F=6} 
  m_imgHue.create(m_imgHSV.size(), m_imgHSV.depth());

  // cv::mixChannels是把输入的矩阵(或矩阵数组)的某些通道拆分复制给对应的输出矩阵(或矩阵数据)的某些通道中
  // 函数原型为 void  mixChannels (const Mat*  src , int  nsrc , Mat*  dst , int  ndst , const int*  fromTo , size_t  npairs );
  // src: 输入矩阵,可以为一个或多个,但矩阵必须有相同的大小和深度
  // nsrc: 输入矩阵的个数
  // ch: 序号对向量, 决定哪个通道被拷贝；偶数下标用来标识输入矩阵, 奇数下标用来标识输出矩阵；若偶数下标为负数,则对应的输出矩阵为零矩阵
  // 所以,这个函数实现从hsv图像中仅拷贝h通道数据
  cv::mixChannels(&m_imgHSV, 1, &m_imgHue, 1, ch, 1);
  imshow("hsv", Tracker::m_imgHSV);
  imshow("hue", Tracker::m_imgHue);

  // Check if time to init
  // 若已选择好物体,初始化跟踪框,提取目标直方图
  if (g_initTracking < 0) {
    InitTrackWindow(img, g_selRect);
  }

  // 已提取好目标直方图
  if (g_initTracking > 0) {
    // 用卡尔曼滤波预测一下帧出现的位置
    Mat prediction = m_KF.predict();

    const float *phranges = m_histRanges;
    // 计算反向投影, 即像素符合目标直方图的概率图
    // m_imgHue:输入当前帧h分量图
    // m_hist是目标的直方图
    // m_imgBackproject:输出的反向投影图
    cv::calcBackProject(&m_imgHue, 1, 0, m_hist, m_imgBackproject, &phranges);
    // 让反向投影图和掩码进行"与"操作,去除边缘等
    m_imgBackproject &= m_imgMask;

    // double bhatt = compareHist(hist, back_hist, CV_COMP_BHATTACHARYYA);
    // cout << "B error " << bhatt << endl;

    // camshift跟踪方法,其实它可以用来跟踪 反向投影图中 表达的任何种类的特征的分布,在这里跟踪h分量也是因为前面计算的是h分量
    // m_trackBox是一个旋转矩形,有三个参数(质心 矩形长宽 旋转角度)
    // ?? but camshift算法返回的是什么呢
    // m_trackWindow是camshift迭代开始的位置
    // camshift函数返回rotatedrect形式的旋转矩阵,会自动调整跟踪物体框尺寸大小
    m_trackBox = cv::CamShift(m_imgBackproject, m_trackWindow,
                              TermCriteria(TermCriteria::EPS | TermCriteria::COUNT, 10, 1));

    // if (m_trackWindow.area() <= 1) {
    //   int cols = m_imgBackproject.cols, rows = m_imgBackproject.rows, r = (MIN(cols, rows) + 5) / 6;
    //   m_trackWindow = Rect(m_trackWindow.x - r, m_trackWindow.y - r,
    //                        m_trackWindow.x + r, m_trackWindow.y + r) &
    //                   Rect(0, 0, cols, rows);
    // }
    // Rect pRect(prediction.at<float>(0), prediction.at<float>(1),m_trackWindow.width,m_trackWindow.height);
    // rectangle(out, pRect, Scalar(0, 0, 255));

    // 用卡尔曼滤波预测位置
    Tracker::PredictPos();
    Tracker::DrawStuff(out);
  }

}

void Tracker::PredictPos()
{
  // 预测目标在下一帧中的位置 用m_trackBox表示
  m_measured.at<float>(0) = m_trackBox.center.x;
  m_measured.at<float>(1) = m_trackBox.center.y;

  m_estimated = m_KF.correct(m_measured);
  cv::Point statePt(m_estimated.at<float>(0), m_estimated.at<float>(1));

  m_past.push_back(m_trackBox.center);
  m_kalmanv.push_back(statePt);

  // m_trackWindow = Rect(statePt.x - m_trackWindow.width / 2, statePt.y - m_trackWindow.height / 2,\
  //  m_trackWindow.width, m_trackWindow.height);

}


void Tracker::DrawStuff(cv::Mat &out)
{

  // 若选择显示反向投影图
  if (m_showBackproject)
    // 把反向投影图从灰度图转换成brg图,赋值给out,这样就不会显示真实的当前帧了
    cvtColor(m_imgBackproject, out, COLOR_GRAY2BGR);

  // trackBox是一个旋转矩阵
  // boundingRect()返回包含旋转矩形的最小外接矩形
  Rect brect = m_trackBox.boundingRect();

  // ellipse(image, m_trackBox, Scalar(0, 255, 255), 3, LINE_AA);
  // btrack是当前帧跟踪到的目标矩阵位置
  Rect btrack(m_trackBox.center.x - brect.width / 2, m_trackBox.center.y - brect.height / 2, brect.width, brect.height);

  // rectangle(image, btrack, Scalar(0, 0, 255));
  // ????? 奇怪m_trackWindow不是仅仅是camshift预测到的位置吗
  // 为什么直接就写了
  rectangle(out, m_trackWindow, Scalar(255, 0, 0));
  // btrack的框更大一些
  rectangle(out, btrack, Scalar(0, 0, 255));

  // 输出卡尔曼滤波预测的运动方向 位置
  // 但为什么最后画的线只有第一帧有呢
  // 这里其实是以线成点吗???
  for (int i = 0; i < m_kalmanv.size() - 1; i++)
    line(out, m_kalmanv[i], m_kalmanv[i + 1], Scalar(0, 0, 255), 3);

  // 最后画的点还是卡尔曼滤波预测的,目标框的中心点
  Point statePt(m_estimated.at<float>(0), m_estimated.at<float>(1));
  drawCross(statePt, Scalar(255, 0, 255), 5);
// drawCross(measPt, Scalar(0, 180, 255), 5);
}

void Tracker::HideControlsGUI()
{
  cvDestroyWindow("Trackbars");
  m_showControlsGUI = false;
}

bool Tracker::ToggleShowBackproject()
{
  
  m_showBackproject = !m_showBackproject;
}

// void Tracker::ShowBackproject()
// {
//   cvNamedWindow("CamShift", 1);
//   if (!m_showBackproject)
//   {
//     cvMoveWindow(m_backprojectWndName.c_str(), (2 * m_frameSize.width) + 20, 0);
//   }
//   cvShowImage(m_backprojectWndName.c_str(), m_imgBackproject);
//   m_showBackproject = true;
// }

// void Tracker::HideBackproject()
// {
//   cvDestroyWindow(m_backprojectWndName.c_str());
//   m_showBackproject = false;
// }

// void Tracker::ShowHistogram()
// {
//   cvNamedWindow(m_histogramWndName.c_str(), 1);
//   if (!m_showHistogram)
//   {
//     cvMoveWindow(m_histogramWndName.c_str(), (2 * m_frameSize.width) + 20, m_frameSize.height + 55);
//   }
//   cvShowImage(m_histogramWndName.c_str(), m_histImg);
//   m_showHistogram = true;
// }

// void Tracker::HideHistogram()
// {
//   cvDestroyWindow(m_histogramWndName.c_str());
//   m_showHistogram = false;
// }

