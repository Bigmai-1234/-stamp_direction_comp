# -stamp_direction_comp

任务描述，
我们需要知道，我们盖在文件上的红章是否是端正的。





需要解决的问题：
1、图章的识别
2、图章的定位
3、图章的方向判定

思路：
图章基本上是红色的，我们先根据颜色提取可能的图章区域。
当然，假如文档中，还有其他红色的区域，这一步都会提取出来。

img = cv2.imdecode(np.fromfile("jingshanshi_muti_stamp.png", dtype=np.uint8), -1)


def extract_red(img):
    ''''使用inRange方法，拼接mask0,mask1'''

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    rows, cols, channels = img.shape
    # 区间1
    lower_red = np.array([0, 43, 46])
    upper_red = np.array([10, 255, 255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)
    # 区间2
    lower_red = np.array([156, 43, 46])
    upper_red = np.array([180, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)
    # 拼接两个区间
    mask = mask0 + mask1
    return mask


mask = extract_red(img)
mask_img = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=mask)
#cv2.imwrite('jingshanshi_muti_stamp_pickred.png', mask_img)



公章一般都是圆形的，我们先利用HoughCircles 找出圆来

# cv2.HoughCircles 寻找出圆，匹配出图章的位置

circles = cv2.HoughCircles(binaryImg, cv2.HOUGH_GRADIENT, 1, 40,
                           param1=50, param2=30, minRadius=20, maxRadius=60)

circles = np.uint16(np.around(circles))

提取出文档中，图形的轮廓。根据点到圆心的距离小于或等于半径，进一步定位公章位置

# findContours(image, mode, method[, contours[, hierarchy[, offset]]]) -> image, contours, hierarchy
binaryImg = cv2.Canny(mask_img, 50, 200)  # 二值化，canny检测
image, contours, hierarchy = cv2.findContours(binaryImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

'''
tuple cir_point:圆心坐标
int radius:半径
list points:点集
float dp: 误差
int samplingtime:采样次数

'''
import random


def circle_check(cir_point, radius, points, dp=4, samplingtime=30):
    # 根据点到圆心的距离等于半径判断点集是否在圆上
    # 多次抽样出5个点
    # 判断多次抽样的结果是否满足条件
    count = 0
    points = list(points)
    for s in range(samplingtime):
        # 从点集points 中采样一次
        points_samp = random.sample(points, 5)
        # 判断点到圆心的距离是否等于半径
        points_samp = np.array(points_samp[0])
        dist = np.linalg.norm(points_samp - cir_point)
        if dist == radius or abs(dist - radius) <= dp:
            continue
        else:
            count += 1
    if count < 3:
        return True
    else:
        return False


def circle_map(contours, circles):
    is_stramp = [0] * len(contours)
    circle_point = []
    for cir in circles[0, :]:
        # 获取圆心和半径
        cir_point = np.array((cir[0], cir[1]))
        radius = cir[2]

        # 遍历每一个点集
        for cidx, cont in enumerate(contours):
            # 当轮廓点数少于10 的时候，默认其不是公章轮廓
            if len(cont) < 10:
                continue
            # 匹配出公章轮廓，并对应出圆心坐标
            stampcheck = circle_check(cir_point, radius, cont, dp=6, samplingtime=40)
            # 如果满足点在圆心上，就将圆心,半径和对应的点记录
            if stampcheck:
                circle_point.append((cir_point, radius, cont))
                is_stramp[cidx] = 1

    return circle_point, is_stramp


circle_point, is_stramp = circle_map(contours, circles)

上面的代码中，用到了采样的方法，如何判断轮廓是圆弧呢？每个像素点穷举显然不合适，于是我们多次采样，然后投票判决。



找到公章的外围轮廓之后，我们进一步得到文字所在的轮廓像素。



获得外围轮廓是很有意义的，去掉他们，将会对后面的定位至关重要。

接下来，我们有圆心坐标，文字轮廓，如何计算公章朝向呢？

整体思路是：

我们对每一个文字轮廓，进行中心点坐标的计算。根据中心点和圆心组成的向量，计算夹角。然后找出满足一定条件的夹角对应的两个向量。再对这两个向量做差后计算与水平轴的夹角。

如下图，上面的方案能很好的找出公章下方的边缘位置：



上面的过程需要解决一些问题：

1、排除多个非常靠近的中心点的干扰，可以用聚类算法，基于角度的计算等等。我用的是第二种方法。

2、中心点分象限处理，分象限是很有必要的，在每个象限里，我们只需要考虑两个边际。







