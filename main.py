from PIL import Image
from numpy import asarray
import numpy as np


def convert32to64():
    m = 0
    n = 0
    image64 = np.array(np.zeros((64, 64, 3), dtype=np.uint8))
    img = Image.open('testpic32.png')
    imgarray = asarray(img)
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
    # print(imgarray[15])
    # print(image64.shape)
    # print((image64))
    img = Image.fromarray(image64)
    img.save('mypic64.png')


# img.show()


def convert64to128():
    m = 0
    n = 0
    image128 = np.array(np.zeros((128, 128, 3), dtype=np.uint8))
    img = Image.open('mypic64.png')
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
    # print(imgarray[15])
    # print(image64.shape)
    # print(image128.shape)
    img = Image.fromarray(image128)
    img.save('mypic128.png')
    # img.show()


def convert128to256():
    m = 0
    n = 0
    image256 = np.array(np.zeros((256, 256, 3), dtype=np.uint8))
    img = Image.open('mypic128.png')
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
    # print(imgarray[15])
    # print(image64.shape)
    print(image256.shape)
    img = Image.fromarray(image256)
    img.save('mypic256.png')
    img.show()


def convertLiner32to127():
    m = 0
    n = 0
    image128 = np.array(np.zeros((128, 128, 3), dtype=np.uint8))
    img = Image.open('testpic32.png')
    imgarray = asarray(img)
    for i in range(0, 128, 4):
        for j in range(0, 128, 4):
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

    ImageResult = caculateliner(image128)
    print(ImageResult.shape)
    print(ImageResult)
    img = Image.fromarray(ImageResult)
    img.save('mypicLiner128.png')
    img.show()


# img.show()


def caculateliner(image128):
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


if __name__ == '__main__':
    # convert32to64()
    # convert64to128()
    # convert128to256()
    convertLiner32to127()
