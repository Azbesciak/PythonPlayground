from __future__ import division
from pylab import *
import skimage as ski
from skimage import data, io, filters, exposure, measure
from skimage.filters import rank
from skimage import img_as_float, img_as_ubyte
from skimage.morphology import disk
import skimage.morphology as mp
from skimage import util
from skimage.color import rgb2hsv, hsv2rgb, rgb2gray
from skimage.filters.edges import convolve
from matplotlib import pylab as plt
import numpy as np
from numpy import array
from IPython.display import display
from ipywidgets import interact, interactive, fixed
from ipywidgets import *
from ipykernel.pylab.backend_inline import flush_figures
from multiprocessing.pool import ThreadPool

planesToRead = [
    '01','02','04',
    '05','06','08',
    '09', '10', '11',
    '12', '13', '14',
    '15', '16', '17',
    '18', '19', '00'
    # '17'
]


def drawPlanesImage(i):
    plt.subplot(6,3,i)


planes = [io.imread('./planes/samolot{0}.jpg'.format(i)) for i in planesToRead]
imageList = []
for plane in planes:
    imageList.append(plane)


def increaseEdges(img, lim = 0.3):
    img = mp.erosion(img)
    img[img[:,:] >= lim] = 1
    img[img[:,:] < lim] = 0
    img = mp.dilation(img)
    img = mp.dilation(img)
    img = mp.erosion(img)

    return img


# def getEdges(img, scale=0.15):
#     K = array([[ 1, 2, 1],
#            [ 0, 0, 0],
#            [-1,-2,-1]])
#     K = K / 4
#
#     K2 = array([[ -1, 0, 1],
#                [ -2, 0, 2],
#                [-1,0,1]])
#     K2 = K2 / 4
#     img = rgb2gray(img)
#     res1 = np.abs(convolve(img, K))
#     res2 = np.abs(convolve(img, K2))
#     res = (res1 + res2) / 2
# #     print(res)
#     return increaseEdges(res, scale)


# def getEdges(image, level = 0.07):
#     blackWhite = rgb2gray(image)
#     blackWhite = blackWhite ** 1.5
#     blackWhite = filters.sobel(blackWhite)
#     blackWhite = blackWhite ** 0.7
#     blackWhite = filters.sobel(blackWhite)
#     blackWhite = mp.dilation(blackWhite)
#     blackWhite = mp.erosion(blackWhite)
#
#     blackWhite[blackWhite[:,:] >= level] = 1
#     blackWhite[blackWhite[:,:] < level] = 0
#     return blackWhite

def getEdges(img, x): #gama 0.7 sig = 4.65
    img = rgb2gray(img)
    from skimage import feature
    img = img ** 0.7
    img = ski.feature.canny(img, sigma=4.65)
    img = mp.dilation(img)
    return img


def processImage(img):
    print("started")
    res = getEdges(img, 0.09)
    res = hsv2rgb(colorImg(res))
    res = addBorders(img, res)
    return res


def replaceAll(objectsMap, oldId, newId):
    objectsMap[objectsMap == oldId] = newId


def findObjects(img):
    maxObjId = 0
    r = 6
    idsMap = [0]
    objectsMap = np.zeros((len(img), len(img[0])))
    for x in range(r, len(img) - r):
        for y in range(r, len(img[x]) - r):
            if img[x][y] > 0:
                total = 0
                for i in range(-r, r + 1):
                    for j in range(-r, r + 1):
                        total += img[x + i][y + j]
                if (total > 0):
                    objId = 0
                    for i in range(-r, r + 1):
                        for j in range(-r, r + 1):
                            if objId > 0 and objId < objectsMap[x + i][y + j]:
                                replaceAll(objectsMap[:x], objId, objectsMap[x + i][y + j])
                                idsMap[int(objId)] = 0
                            objId = max(objId, objectsMap[x + i][y + j])

                    if (objId > 0):
                        objectsMap[x][y] = objId
                        idsMap[int(objId)] += 1
                    else:
                        maxObjId += 1
                        idsMap.append(1)
                        objectsMap[x][y] = maxObjId
    lev = max(idsMap) / 7
    objCords = []

    for i in range(len(idsMap)):
        objCords.append({
            "left": len(img),
            "right": 0,
            "top": len(img[0]),
            "bottom": 0
            # "x":0, "y": 0, "n":0
        })
    for i in range(len(objectsMap)):
        for j in range(len(objectsMap[i])):
            objId = int(objectsMap[i][j])
            if idsMap[objId] < lev:
                objectsMap[i][j] = 0
                objCords[objId] = 0
            else:
                objCords[objId]["left"] = min(objCords[objId]["left"], i)
                objCords[objId]["right"] = max(objCords[objId]["right"], i)
                objCords[objId]["top"] = min(objCords[objId]["top"], j)
                objCords[objId]["bottom"] = max(objCords[objId]["bottom"], j)
                # objCords[objId]["x"] += i
                # objCords[objId]["y"] += j
                # objCords[objId]["n"] += 1

    for i in range(len(objCords)):
        if objCords[i] != 0:
            x = int((objCords[i]["left"] + objCords[i]["right"]) / 2)
            y = int((objCords[i]["top"] + objCords[i]["bottom"]) / 2)
            # x = int((objCords[i]["x"] / objCords[i]["n"]))
            # y = int((objCords[i]["y"] / objCords[i]["n"]))
            for j in range(-r, r + 1):
                for k in range(-r, r + 1):
                    if x+j < len(objectsMap) and x+j >=0 and y+k < len(objectsMap[x]) and y+k >= 0:
                        objectsMap[x+j][y+k] = idsMap[i]

    return objectsMap, maxObjId


def colorImg(img):
    objectsMap, maxObjId = findObjects(img)
    colorStep = 360 / maxObjId
    colorsMap = []
    for i in range(len(img)):
        colorsMap.append([])
        for j in range(len(img[i])):
            h = colorStep * objectsMap[i][j]
            if not h == 0:
                colorsMap[i].append([h, 1, 1])
            else:
                colorsMap[i].append([0, 0, 0])
    return colorsMap


def addBorders(original, borders):
    for i in range(len(original)):
        for j in range(len(original[i])):
            if sum(borders[i][j]) > 0.1:
                original[i][j] = borders[i][j] * 255
    return original


fig = plt.figure(facecolor="black", figsize=(60, 60))  # czarne tło


def fun(p=1, k=20):
    elements = len(imageList)
    imagesProc = []
    pool = ThreadPool(processes=elements)
    count = 0
    for img in imageList:
        count += 1
        asyncRes = pool.apply_async(processImage, ([img]))
        imagesProc.append(asyncRes)
        print(count)

    for i in range(len(imagesProc)):
        print("waiting for {0}".format(i))
        res = imagesProc[i].get()
        #         print(res)
        drawPlanesImage(i + 1)
        plt.imshow(res)
    plt.tight_layout()  # Aby obrazy znajdowały się obok siebie
    plt.show()
    fig.savefig("samoloty4.pdf")
    plt.close()


fun()