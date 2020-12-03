# define Library
from PIL import Image
from numpy import asarray
import numpy as np
import cv2
import math
import sys
import matplotlib.pyplot as plt

datapot = np.array(np.zeros((3, 6)))


#               Convert Image 32*32 in to 128*128       *4
def convert32to128(selecter):
    if (selecter == 1):

        m = 0
        n = 0
        image64 = np.array(np.zeros((64, 64, 3), dtype=np.uint8))  # define list size 64 for save matrix image
        #              read image and set in array
        img = Image.open('myImageDec32.png')
        imgarray = asarray(img)
        #              Reduplication item
        for i in range(0, 64, 2):
            for j in range(0, 64, 2):
                image64[i][j][0] = imgarray[m][n][0]
                image64[i][j][1] = imgarray[m][n][1]
                image64[i][j][2] = imgarray[m][n][2]
                # next
                image64[i][j + 1][0] = imgarray[m][n][0]
                image64[i][j + 1][1] = imgarray[m][n][1]
                image64[i][j + 1][2] = imgarray[m][n][2]
                # next
                image64[i + 1][j + 1][0] = imgarray[m][n][0]
                image64[i + 1][j + 1][1] = imgarray[m][n][1]
                image64[i + 1][j + 1][2] = imgarray[m][n][2]
                # next
                image64[i + 1][j][0] = imgarray[m][n][0]
                image64[i + 1][j][1] = imgarray[m][n][1]
                image64[i + 1][j][2] = imgarray[m][n][2]

                n = n + 1
            n = 0
            m += 1

        img = Image.fromarray(image64)
        img.save('myImage64.png')
        img.show()
    elif (selecter == 2):
        m = 0
        n = 0
        image128 = np.array(np.zeros((128, 128, 3), dtype=np.uint8))
        img = Image.open('myImage64.png')
        imgarray = asarray(img)
        for i in range(0, 128, 2):
            for j in range(0, 128, 2):
                image128[i][j][0] = imgarray[m][n][0]
                image128[i][j][1] = imgarray[m][n][1]
                image128[i][j][2] = imgarray[m][n][2]
                # next
                image128[i][j + 1][0] = imgarray[m][n][0]
                image128[i][j + 1][1] = imgarray[m][n][1]
                image128[i][j + 1][2] = imgarray[m][n][2]
                # next
                image128[i + 1][j + 1][0] = imgarray[m][n][0]
                image128[i + 1][j + 1][1] = imgarray[m][n][1]
                image128[i + 1][j + 1][2] = imgarray[m][n][2]
                # next
                image128[i + 1][j][0] = imgarray[m][n][0]
                image128[i + 1][j][1] = imgarray[m][n][1]
                image128[i + 1][j][2] = imgarray[m][n][2]

                n = n + 1
            n = 0
            m += 1

        img = Image.fromarray(image128)
        img.save('myImage128.png')
        img.show()

        #           In this section, the function enlarges the  image


def convert128to256():
    m = 0
    n = 0
    image256 = np.array(np.zeros((256, 256, 3), dtype=np.uint8))

    img = Image.open('myImage128.png')
    imgarray = asarray(img)
    for i in range(0, 256, 2):
        for j in range(0, 256, 2):
            image256[i][j][0] = imgarray[m][n][0]
            image256[i][j][1] = imgarray[m][n][1]
            image256[i][j][2] = imgarray[m][n][2]
            # next
            image256[i][j + 1][0] = imgarray[m][n][0]
            image256[i][j + 1][1] = imgarray[m][n][1]
            image256[i][j + 1][2] = imgarray[m][n][2]
            # next
            image256[i + 1][j + 1][0] = imgarray[m][n][0]
            image256[i + 1][j + 1][1] = imgarray[m][n][1]
            image256[i + 1][j + 1][2] = imgarray[m][n][2]
            # next
            image256[i + 1][j][0] = imgarray[m][n][0]
            image256[i + 1][j][1] = imgarray[m][n][1]
            image256[i + 1][j][2] = imgarray[m][n][2]

            n = n + 1
        n = 0
        m += 1

    img = Image.fromarray(image256)
    img.save('myImage256.png')
    img.show()

    #           Linear interpolation part to do


def convertLinearInterpolation32to128():
    m = 0
    n = 0
    image128 = np.array(np.zeros((128, 128, 3), dtype=np.uint8))
    img = Image.open('myImageDec32.png')
    imgarray = asarray(img)

    #          Row Calculation   Linear interpolation
    for i in range(0, 128, 4):
        for j in range(0, 128, 4):
            #                   Set  item in 128*128 array
            image128[i][j][0] = imgarray[m][n][0]
            image128[i][j][1] = imgarray[m][n][1]
            image128[i][j][2] = imgarray[m][n][2]
            if (j == 124):
                image128[i][j + 3][0] = imgarray[m][n][0]
                image128[i][j + 3][1] = imgarray[m][n][1]
                image128[i][j + 3][2] = imgarray[m][n][2]

            if (i == 124):
                for j in range(0, 128, 4):
                    image128[i + 3][j][0] = imgarray[m][n][0]
                    image128[i + 3][j][1] = imgarray[m][n][1]
                    image128[i + 3][j][2] = imgarray[m][n][2]
                    if (j == 124):
                        image128[i + 3][j + 3][0] = imgarray[m][n][0]
                        image128[i + 3][j + 3][1] = imgarray[m][n][1]
                        image128[i + 3][j + 3][2] = imgarray[m][n][2]
            n += 1

        n = 0
        m += 1
        #                   Calculation   Linear interpolation
    ImageResult = caculateliner(image128)

    img = Image.fromarray(ImageResult)
    img.save('mypicLiner128.png')
    img.show()


#                   Calculation   Linear interpolation in this below function
def caculateliner(image128):
    # Row Calculation   Linear interpolation
    for i in range(0, 128, 4):
        for j in range(0, 128, 4):
            if (j != 124):
                result1, result2, result3 = caculatelineritem(image128[i][j][0], image128[i][j + 4][0])
                image128[i][j + 1][0] = int(result1)
                image128[i][j + 1][1] = int(result1)
                image128[i][j + 1][2] = int(result1)
                # next
                image128[i][j + 2][0] = int(result2)
                image128[i][j + 2][1] = int(result2)
                image128[i][j + 2][2] = int(result2)
                # next
                image128[i][j + 3][0] = int(result3)
                image128[i][j + 3][1] = int(result3)
                image128[i][j + 3][2] = int(result3)

            else:
                result1, result2 = caculatelineritem(image128[i][j][0], image128[i][j + 3][0], True)

                image128[i][j + 1][0] = int(result1)
                image128[i][j + 1][1] = int(result1)
                image128[i][j + 1][2] = int(result1)

                # next
                image128[i][j + 2][0] = int(result2)
                image128[i][j + 2][1] = int(result2)
                image128[i][j + 2][2] = int(result2)

    for j in range(0, 128, 4):
        if (j != 124):
            result1, result2, result3 = caculatelineritem(image128[127][j][0], image128[127][j + 4][0])
            image128[127][j + 1][0] = int(result1)
            image128[127][j + 1][1] = int(result1)
            image128[127][j + 1][2] = int(result1)
            # next
            image128[127][j + 2][0] = int(result2)
            image128[127][j + 2][1] = int(result2)
            image128[127][j + 2][2] = int(result2)
            # next
            image128[127][j + 3][0] = int(result3)
            image128[127][j + 3][1] = int(result3)
            image128[127][j + 3][2] = int(result3)
        else:
            result1, result2 = caculatelineritem(image128[127][j][0], image128[127][j + 3][0], True)

            image128[127][j + 1][0] = int(result1)
            image128[127][j + 1][1] = int(result1)
            image128[127][j + 1][2] = int(result1)

            # next
            image128[127][j + 2][0] = int(result2)
            image128[127][j + 2][1] = int(result2)
            image128[127][j + 2][2] = int(result2)

            # colom Calculation   Linear interpolation
    for j in range(0, 128, 1):
        for i in range(0, 128, 4):
            if (i != 124):
                result1, result2, result3 = caculatelineritem(image128[i][j][0], image128[i + 4][j][0])
                image128[i + 1][j][0] = int(result1)
                image128[i + 1][j][1] = int(result1)
                image128[i + 1][j][2] = int(result1)
                # next
                image128[i + 2][j][0] = int(result2)
                image128[i + 2][j][1] = int(result2)
                image128[i + 2][j][2] = int(result2)
                # next
                image128[i + 3][j][0] = int(result3)
                image128[i + 3][j][1] = int(result3)
                image128[i + 3][j][2] = int(result3)

            else:
                result1, result2 = caculatelineritem(image128[i][j][0], image128[i + 3][j][0], True)

                image128[i + 1][j][0] = int(result1)
                image128[i + 1][j][1] = int(result1)
                image128[i + 1][j][2] = int(result1)
                # next
                image128[i + 2][j][0] = int(result2)
                image128[i + 2][j][1] = int(result2)
                image128[i + 2][j][2] = int(result2)
    for i in range(0, 128, 4):
        if (i != 124):
            result1, result2, result3 = caculatelineritem(image128[i][127][0], image128[i + 4][127][0])
            image128[i + 1][127][0] = int(result1)
            image128[i + 1][127][1] = int(result1)
            image128[i + 1][127][2] = int(result1)
            # next
            image128[i + 2][127][0] = int(result2)
            image128[i + 2][127][1] = int(result2)
            image128[i + 2][127][2] = int(result2)
            # next
            image128[i + 3][127][0] = int(result3)
            image128[i + 3][127][1] = int(result3)
            image128[i + 3][127][2] = int(result3)

        else:
            result1, result2 = caculatelineritem(image128[i][127][0], image128[i + 3][127][0], True)

            image128[i + 1][127][0] = int(result1)
            image128[i + 1][127][1] = int(result1)
            image128[i + 1][127][2] = int(result1)
            # next
            image128[i + 2][127][0] = int(result2)
            image128[i + 2][127][1] = int(result2)
            image128[i + 2][127][2] = int(result2)

    return image128

    #        Take Two point and add in  formula


def caculatelineritem(point1, point2, islast=None):
    if (islast != True):
        result1 = (point2 * 1) / 4 + (point1 * 3) / 4
        result2 = (point2 * 2) / 4 + (point1 * 2) / 4
        result3 = (point2 * 3) / 4 + (point1 * 1) / 4
        return result1, result2, result3
    else:
        result1 = (point2 * 1) / 3 + (point1 * 2) / 3
        result2 = (point2 * 2) / 3 + (point1 * 1) / 3
        return result1, result2

        # Get the error rate of the algorithm


def evalutionlinerfunction(index,imagename):
    img1 = Image.open('mypicLiner128.png')
    img2 = Image.open(imagename)
    IMG = cv2.imread('myImageDec32.png')
    img3 = cv2.resize(IMG, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
    imgarray1 = asarray(img1)
    imgarray2 = asarray(img2)
    imgarray3 = asarray(img3)
    eval1 = caculated(imgarray1, imgarray2)
    eval2 = caculated(imgarray3, imgarray2)

    datapot[index][2] = eval1
    datapot[index][3] = eval2
    # Get the error rate of the algorithm nearest


def evalutionNearstFunction(index,imagname):
    img1 = Image.open('myImage128.png')
    img2 = Image.open(imagname)
    IMG = cv2.imread('myImageDec32.png')
    img3 = cv2.resize(IMG, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    imgarray1 = asarray(img1)
    imgarray2 = asarray(img2)
    print(imgarray2.shape)
    print(imgarray1.shape)
    imgarray3 = asarray(img3)
    eval1 = caculated(imgarray1, imgarray2)
    eval2 = caculated(imgarray3, imgarray2)
    datapot[index][0]=eval1
    datapot[index][1] = eval2

    #       Calculation function


def caculated(list1, list2):
    result = np.array(np.zeros((128, 128, 3), dtype=np.uint8))
    res = np.subtract(list2, list1)
    return np.average(res) / 3


#           Decrease orginal image
def decreaseimageto32(nameimage):
    n = 0
    m = 0
    img2 = Image.open(nameimage)
    imgarray2 = asarray(img2)

    image32 = np.array(np.zeros((32, 32, 3), dtype=np.uint8))
    for i in range(0, 128, 4):
        for j in range(0, 128, 4):
            image32[m][n] = imgarray2[i][j]
            n += 1
        n = 0
        m += 1
    img = Image.fromarray(image32)
    img.save('myImageDec32.png')
    img.show()

    # First, get the weights according to the location


def u(s, a):
    if (abs(s) >= 0) & (abs(s) <= 1):
        return (a + 2) * (abs(s) ** 3) - (a + 3) * (abs(s) ** 2) + 1
    elif (abs(s) > 1) & (abs(s) <= 2):
        return a * (abs(s) ** 3) - (5 * a) * (abs(s) ** 2) + (8 * a) * abs(s) - 4 * a
    return 0

    # In this section, we create 4x4 presentation values  and get each point according to the other points.


def SegmentImage(img, H, W, C):
    zimg = np.zeros((H + 4, W + 4, C))
    zimg[2:H + 2, 2:W + 2, :C] = img
    #  the firsttwo col/last  and row
    zimg[2:H + 2, 0:2, :C] = img[:, 0:1, :C]
    zimg[H + 2:H + 4, 2:W + 2, :] = img[H - 1:H, :, :]
    zimg[2:H + 2, W + 2:W + 4, :] = img[:, W - 1:W, :]
    zimg[0:2, 2:W + 2, :C] = img[0:1, :, :C]
    # Pad the missing eight points
    zimg[0:2, 0:2, :C] = img[0, 0, :C]
    zimg[H + 2:H + 4, 0:2, :C] = img[H - 1, 0, :C]
    zimg[H + 2:H + 4, W + 2:W + 4, :C] = img[H - 1, W - 1, :C]
    zimg[0:2, W + 2:W + 4, :C] = img[0, W - 1, :C]
    return zimg


def get_progressbar_str(progress):
    END = 170
    MAX_LEN = 30
    BAR_LEN = int(MAX_LEN * progress)
    return ('Progress:[' + '=' * BAR_LEN +
            ('>' if BAR_LEN < MAX_LEN else '') +
            ' ' * (MAX_LEN - BAR_LEN) +
            '] %.1f%%' % (progress * 100.))


# Bicubic operation
def bicubic(img, ratio, a):
    # Get image size
    H, W, C = img.shape

    img = SegmentImage(img, H, W, C)
    # Create new image
    dH = math.floor(H * ratio)
    dW = math.floor(W * ratio)
    dst = np.zeros((dH, dW, 3))

    h = 1 / ratio

    print('Start bicubic interpolation')
    print('It will take a little while...')
    inc = 0
    for c in range(C):
        for j in range(dH):
            for i in range(dW):
                x, y = i * h + 2, j * h + 2

                x1 = 1 + x - math.floor(x)
                x2 = x - math.floor(x)
                x3 = math.floor(x) + 1 - x
                x4 = math.floor(x) + 2 - x

                y1 = 1 + y - math.floor(y)
                y2 = y - math.floor(y)
                y3 = math.floor(y) + 1 - y
                y4 = math.floor(y) + 2 - y

                mat_l = np.matrix([[u(x1, a), u(x2, a), u(x3, a), u(x4, a)]])
                mat_m = np.matrix([[img[int(y - y1), int(x - x1), c], img[int(y - y2), int(x - x1), c],
                                    img[int(y + y3), int(x - x1), c], img[int(y + y4), int(x - x1), c]],
                                   [img[int(y - y1), int(x - x2), c], img[int(y - y2), int(x - x2), c],
                                    img[int(y + y3), int(x - x2), c], img[int(y + y4), int(x - x2), c]],
                                   [img[int(y - y1), int(x + x3), c], img[int(y - y2), int(x + x3), c],
                                    img[int(y + y3), int(x + x3), c], img[int(y + y4), int(x + x3), c]],
                                   [img[int(y - y1), int(x + x4), c], img[int(y - y2), int(x + x4), c],
                                    img[int(y + y3), int(x + x4), c], img[int(y + y4), int(x + x4), c]]])
                mat_r = np.matrix([[u(y1, a)], [u(y2, a)], [u(y3, a)], [u(y4, a)]])
                dst[j, i, c] = np.dot(np.dot(mat_l, mat_m), mat_r)

                # Print progress
                inc = inc + 1
                sys.stderr.write('\r\033[K' + get_progressbar_str(inc / (C * dH * dW)))
                sys.stderr.flush()
    sys.stderr.write('\n')
    sys.stderr.flush()
    return dst


#

def convertbicubic32to128():
    img = cv2.imread('myImageDec32.png')

    # Scale factor
    ratio = 2
    # Coefficient
    a = -1 / 2

    dst = bicubic(img, ratio, a)
    dst2 = bicubic(dst, ratio, a)
    cv2.imwrite('bicubic.png', dst2)


def evaluatedbicubic23to128(index,imagename):
    img1 = Image.open('bicubic.png')
    img2 = Image.open(imagename)
    IMG = cv2.imread('myImageDec32.png')
    img3 = cv2.resize(IMG, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    imgarray1 = asarray(img1)
    imgarray2 = asarray(img2)
    imgarray3 = asarray(img3)
    eval1 = caculated(imgarray1, imgarray2)
    eval2 = caculated(imgarray3, imgarray2)
    datapot[index][4] = eval1
    datapot[index][5] = eval2

def nearest_neighbor():
    convert32to128(1)
    convert32to128(2)

def MyPlot(array):
    activities = ['Mynearest neighbor interpolation', 'Open CV nearest neighbor interpolation', 'My linear interpolation', 'Open Cv linear interpolation','My bicubic interpolation', 'Open Cv bicubic interpolation']

    # portion covered by each label


    # color for each label
    colors = ['r', 'y', 'g', 'b',"orange","purple"]

    # plotting the pie chart
    for i in range(0,3,1):
        plt.subplot(3,1,i+1)
        print(datapot[i][:])
        plt.pie(datapot[i][:], labels=activities, colors=colors,
                startangle=90, shadow=True, explode=(0.2, 0, 0.1, 0, 0.1, 0.2),
                radius=1.2, autopct='%1.1f%%')

        # plotting legend
        plt.title(array[i])

    # showing the plot
    plt.show()
if __name__ == '__main__':
    arrayimage = np.array(['Lenna.png', 'peppers.png', 'Pan128.png'])


    for item in range(0,3,1):
        decreaseimageto32(arrayimage[item])
        nearest_neighbor()
        convert128to256()
        convertLinearInterpolation32to128()
        convertbicubic32to128()
        evalutionNearstFunction(item,arrayimage[item])
        evalutionlinerfunction(item,arrayimage[item])
        evaluatedbicubic23to128(item,arrayimage[item])
        print("Complated: ",arrayimage[item])
    MyPlot(arrayimage)