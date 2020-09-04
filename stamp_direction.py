#!/usr/bin/env python
# coding: utf-8

# （minAreaRect()）来操作，该方法会返回最小外接矩形的中心点左边，
#   矩形宽度、高度，以及旋转角度。因为图像中只有一个文字，
#   因此包含该文字的最小外接矩形返回的角度就是图像的旋转角度（当然也有可能是负值）。

# 

# 
# 
# <!-- findContours( InputOutputArray image, OutputArrayOfArrays contours,
#                               OutputArray hierarchy, int mode,
#                               int method, Point offset=Point());
#  -->
# 

# ```
# findContours( InputOutputArray image, OutputArrayOfArrays contours,
#                               OutputArray hierarchy, int mode,>
#                               int method, Point offset=Point());
#                               
# 
# image 一般是经过Canny、拉普拉斯等边缘检测算子处理过的二值图像；
# contours，向量内每个元素保存了一组由连续的Point点构成的点的集合的向量，每一组Point点集就是一个轮廓。  有多少轮廓，向量contours就有多少元素。
# hierarchy向量内每一个元素的4个int型变量——hierarchy[i][0] ~hierarchy[i][3]，分别表示第i个轮廓的后一个轮廓、前一个轮廓、父轮廓、内嵌轮廓的索引编号
# mode 取值一：CV_RETR_EXTERNAL只检测最外围轮廓，包含在外围轮廓内的内围轮廓被忽略
# 
# 
# 
#            取值二：CV_RETR_LIST   检测所有的轮廓，包括内围、外围轮廓，但是检测到的轮廓不建立等级关
# 
#                   系，彼此之间独立，没有等级关系，这就意味着这个检索模式下不存在父轮廓或内嵌轮廓，
# 
#                   所以hierarchy向量内所有元素的第3、第4个分量都会被置为-1，具体下文会讲到
# 
# 
# 
#            取值三：CV_RETR_CCOMP  检测所有的轮廓，但所有轮廓只建立两个等级关系，外围为顶层，若外围
# 
#                   内的内围轮廓还包含了其他的轮廓信息，则内围内的所有轮廓均归属于顶层
# 
# 
# 
#            取值四：CV_RETR_TREE， 检测所有轮廓，所有轮廓建立一个等级树结构。外层轮廓包含内层轮廓，内
# 
#                    层轮廓还可以继续包含内嵌轮廓。
# 
# method，定义轮廓的近似方法：
# 
#            取值一：CV_CHAIN_APPROX_NONE 保存物体边界上所有连续的轮廓点到contours向量内
# 
#            取值二：CV_CHAIN_APPROX_SIMPLE 仅保存轮廓的拐点信息，把所有轮廓拐点处的点保存入contours
# 
#                    向量内，拐点与拐点之间直线段上的信息点不予保留
# 
#            取值三和四：CV_CHAIN_APPROX_TC89_L1，CV_CHAIN_APPROX_TC89_KCOS使用teh-Chinl chain 近
# 
#                    似算法
# 
# 
# 第六个参数：Point偏移量，所有的轮廓信息相对于原始图像对应点的偏移量，相当于在每一个检测出的轮廓点上加
# 
#             上该偏移量，并且Point还可以是负值！
# 
# ```

# # 如何定位公章
# # 圆的，红的
# # 外圆弧不一定清晰，但一定存在
# # 基本上是红色
# # 一定有字，或有中心图像
# 

# In[6]:


# 此代码提取公章
# 先将红色部分选取，将其他颜色转白
# 对公章单独框定


# import cv2
# import numpy as np
# src = cv2.imread(r"jingshanshi.png")#这里填你的原图像路径
# cv2.namedWindow("input", cv2.WINDOW_AUTOSIZE)
# cv2.imshow("input", src)
# """
# 提取图中的蓝色部分
# """
# hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)#BGR转HSV
# low_hsv = np.array([100,43,46])#这里要根据HSV表对应，填入三个min值（表在下面）
# high_hsv = np.array([124,255,255])#这里填入三个max值
# mask = cv2.inRange(hsv,lowerb=low_hsv,upperb=high_hsv)#提取掩膜

# #黑色背景转透明部分
# mask_contrary = mask.copy()
# mask_contrary[mask_contrary==0]=1
# mask_contrary[mask_contrary==255]=0#把黑色背景转白色
# mask_bool = mask_contrary.astype(bool)
# mask_img = cv2.add(src, np.zeros(np.shape(src), dtype=np.uint8), mask=mask)
# #这个是把掩模图和原图进行叠加，获得原图上掩模图位置的区域
# mask_img=cv2.cvtColor(mask_img,cv2.COLOR_BGR2BGRA)
# mask_img[mask_bool]=[0,0,0,0]
# #这里如果背景本身就是白色，可以不需要这个操作，或者不需要转成透明背景就不需要这里的操作


# cv2.imshow("image",mask_img)
# cv2.imwrite('label123.png',mask_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# In[9]:


import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

"""
提取图中的红色部分
"""

img = cv2.imdecode(np.fromfile("jingshanshi_muti_stamp.png", dtype=np.uint8), -1)


def extract_red(img):
    ''''method1：使用inRange方法，拼接mask0,mask1'''

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
cv2.imwrite('jingshanshi_muti_stamp_pickred.png', mask_img)

######################################提取轮廓#########################################

# findContours(image, mode, method[, contours[, hierarchy[, offset]]]) -> image, contours, hierarchy
binaryImg = cv2.Canny(mask_img, 50, 200)  # 二值化，canny检测
image, contours, hierarchy = cv2.findContours(binaryImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

######################################找圆#########################################

# 判断轮廓是不是正圆
# HoughCircles?
# 判断轮廓中有没有空白

# cv2.HoughCircles 寻找出圆，匹配出图章的位置

circles = cv2.HoughCircles(binaryImg, cv2.HOUGH_GRADIENT, 1, 40,
                           param1=50, param2=30, minRadius=20, maxRadius=60)

circles = np.uint16(np.around(circles))

######################################匹配圆与点集#########################################
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


######################################计算文字区域和旋转角度#########################################

# 把点分成象限
def point2quadrant(points, cc):
    cx, cy = cc[0], cc[1]
    qua1 = []
    qua2 = []
    qua3 = []
    qua4 = []
    for pot in points:
        px, py = pot[0][0], pot[0][1]
        if px >= cx and py >= cy: qua1.append(pot)
        if px < cx and py > cy: qua2.append(pot)
        if px < cx and py < cy:
            qua3.append(pot)
        else:
            qua4.append(pot)
    return qua1, qua2, qua3, qua4


def cos_angle(x, y):
    if (len(x) != len(y)):
        print('error input,x and y is not in the same space')
        return;
    result1 = 0.0;
    result2 = 0.0;
    result3 = 0.0;
    for i in range(len(x)):
        result1 += x[i] * y[i]  # sum(X*Y)
        result2 += x[i] ** 2  # sum(X*X)
        result3 += y[i] ** 2  # sum(Y*Y)
    cosvalue = result1 / ((result2 * result3) ** 0.5)
    return int(math.acos(cosvalue) * 180 / math.pi)


# 计算最大夹角和最小夹角
def angle_comp(v):
    dx = v[0]
    dy = v[1]
    angle = math.atan2(dy, dx)
    angle = int(angle * 180 / math.pi)
    return angle


def max_min_vec(quavec, flag):
    tmp = np.array([])

    if flag == 0:
        ang = 360
        for vec in quavec:
            angt = abs(angle_comp(vec[0]))
            if angt < ang:
                ang = angt
                tmp = vec[0]
    elif flag == 1:
        ang = 0
        # print(quavec)
        for vec in quavec:
            # print(vec)
            angt = abs(angle_comp(vec[0]))
            if angt > ang:
                ang = angt
                tmp = vec[0]
    return tmp


def find_angle(points, cc, angledp):
    # 判断象限
    qua1, qua2, qua3, qua4 = point2quadrant(points, cc)
    qua1vec, qua2vec, qua3vec, qua4vec = np.array(qua1), np.array(qua2), np.array(qua3), np.array(qua4)
    y_posi = np.array([0, cc[1]])
    # print(x_posi)

    # 获得每个象限最大和最小的vec
    veclist = []
    if len(qua1) != 0:
        qua1vec = qua1vec - cc
        veclist.append(max_min_vec(qua1vec, 0))
        veclist.append(max_min_vec(qua1vec, 1))

    if len(qua2) != 0:
        qua2vec = qua2vec - cc
        veclist.append(max_min_vec(qua2vec, 0))
        veclist.append(max_min_vec(qua2vec, 1))

    if len(qua3) != 0:
        qua3vec = qua3vec - cc
        veclist.append(max_min_vec(qua3vec, 0))
        veclist.append(max_min_vec(qua3vec, 1))

    if len(qua4) != 0:
        qua4vec = qua4vec - cc
        veclist.append(max_min_vec(qua4vec, 0))
        veclist.append(max_min_vec(qua4vec, 1))

    # 两两求余弦取最小
    maxa = np.array([])
    maxb = np.array([])
    #     minangle = 0
    #     for a in veclist:
    #         for b in veclist:
    #             cosang = cos_angle(a, b)
    #             if cosang > angledp and cosang < angledp+90:
    #                 if cosang > minangle:
    #                     minangle = cosang
    #                     maxa = a
    #                     maxb = b

    # 依次遍历vec数组，当夹角满足大于阈值和小于阈值时，满足条件
    # 挑选最大夹角对应向量
    maxangle = 0

    veclist2 = veclist[1:]
    veclist2.append(veclist[0])



    for idx, vec in enumerate(veclist):
        cosang = cos_angle(vec, veclist2[idx])
        if cosang > angledp and cosang < angledp + 90:
            if cosang > maxangle:
                minangle = cosang
                maxa = vec
                maxb = veclist2[idx]

    # 计算所得向量与y的夹角
    angle = cos_angle(maxa - maxb, y_posi)
    return angle, maxa, maxb


'''
输入：
圆心，半径
其他不属于圆的轮廓
算法：
计算每一个轮廓中心
对于每一个圆心
    每个轮廓中心到圆心距离小于半径的点
    每一个轮廓中的点按照y坐标升序
    计算旋转角度
输出：
[圆心，半径，文字对应轮廓的像素点，文字对应轮廓中心]
'''


def word_pixel_get(circles, is_stramp, contours):
    # 计算轮廓的中心
    contours_center = np.array([sum(cont) / len(cont) for cont in contours])
    ##
    res = []

    # 遍历每一个圆心
    for cir in circles[0, :]:
        # 创建list搜集满足条件的轮廓，用于可视化
        word_around_px = []
        # 创建list搜集满足条件的轮廓中心，用于计算角度
        word_around_center = []

        # 获取圆心和半径
        cir_point = np.array((cir[0], cir[1]))
        radius = cir[2]
        # 遍历每一个轮廓中心
        for cidx, cc in enumerate(contours_center):
            # 如果是图章外轮廓，则不考虑,因为外轮廓不一定是连续的
            if is_stramp[cidx] == 1: continue
            dist = np.linalg.norm(cc - cir_point)
            # 判断轮廓是否是文字轮廓
            if dist < radius - 3 and dist > radius - 10:
                # 如果是，就搜集起来像素，用于可视化
                word_around_px.append(contours[cidx])
                # 如果是，就搜集中心，用于计算角度
                word_around_center.append(cc)

        res.append([cir_point, radius, word_around_px, word_around_center])
    return res


word_pix = word_pixel_get(circles, is_stramp, contours)


def angle_predict(word_pix):
    # 统计公章数量
    stamps_num = 0
    # 记录公章圆心
    st_cir = []
    # 记录公章旋转角度
    st_rot = []
    # 记录边缘点
    edge_point = []

    for idx, wp in enumerate(word_pix):
        if len(wp[-1]) == 0: continue
        stamps_num += 1
        # 圆心点
        st_cir.append(wp[0])
        # 计算角度
        angle_, maxa, maxb = find_angle(wp[-1], wp[0], 45)

        st_rot.append(angle_)
        edge_point.append(maxa + wp[0])
        edge_point.append(maxb + wp[0])
    return stamps_num, st_cir, st_rot, edge_point


stamps_num, st_cir, st_rot, edge_point = angle_predict(word_pix)

######################################打印结果#########################################
print("疑似图章 {} 个".format(stamps_num))

for st in range(stamps_num):
    print("图章中心 ： ", st_cir[st])
    ang = st_rot[st]
    if ang > 90:
        print("图章右转 ： {} 度".format(ang - 90))
    else:
        print("图章左旋 ： {} 度".format(90 - ang))

