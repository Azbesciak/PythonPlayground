from __future__ import division

from multiprocessing.pool import ThreadPool

import skimage as ski
import skimage.morphology as mp
from matplotlib import pylab as plt
from pylab import *
from skimage import io
from skimage.color import hsv2rgb, rgb2gray

GAMMA = 0.7
SIGMA = 4.65
detection_range = 12
center_circle_range = 5

planesToRead = [
    '01', '02', '04',
    '05', '06', '08',
    '09', '10', '11',
    '12', '13', '14',
    '15', '16', '17',
    '18', '19', '00'
    # '19'
]


def drawPlanesImage(i, img):
    plt.subplot(6, 3, i)
    # plt.subplot(1, 1, i)
    frame = plt.gca()  # Frame do usuniecia osi
    frame.axes.get_xaxis().set_visible(False)  # Usuniecie osix
    frame.axes.get_yaxis().set_visible(False)  # Usuniecie osiy
    plt.imshow(img)


planes = [io.imread('./planes/samolot{0}.jpg'.format(i)) for i in planesToRead]
imageList = []
for plane in planes:
    imageList.append(plane)


def getEdges(img):
    img = rgb2gray(img)
    from skimage import feature
    img = img ** GAMMA
    img = ski.feature.canny(img, sigma=SIGMA)
    img = mp.dilation(img)
    return img


def processImage(img):
    print("started")
    res = getEdges(img)
    res = hsv2rgb(colorImg(res))
    res = addBorders(img, res)
    return res


def replaceAll(objectsMap, oldId, newId):
    objectsMap[objectsMap == oldId] = newId
    return objectsMap


def findObjects(img):
    idsMap, objectsMap, maxObjId = find_objects(img)
    set_center_points(idsMap, objectsMap)
    return objectsMap, maxObjId


def find_objects(img):
    r = detection_range
    maxObjId = 0
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

    idsMap, maxObjId = rescale_ids(idsMap, objectsMap)
    return idsMap, objectsMap, maxObjId


def rescale_ids(idsMap, objectsMap):
    rescaled_id = [0]
    current_id = 0
    for i in range(len(idsMap)):
        if idsMap[i] > 0:
            current_id += 1
            replaceAll(objectsMap, i, current_id)
            rescaled_id.append(idsMap[i])
    objectsMap[objectsMap > current_id] = 0
    maxObjId = len(rescaled_id)
    return rescaled_id, maxObjId


def set_center_points(idsMap, objectsMap):
    lev = max(idsMap) / 7
    objCords = initialize_ids_map(idsMap)
    count_bounding_box(idsMap, lev, objCords, objectsMap)
    measure_center(objCords, objectsMap)


def initialize_ids_map(idsMap):
    objCords = []
    for i in range(len(idsMap)):
        objCords.append({
            "x": 0, "y": 0, "n": 0
        })
    return objCords


def measure_center(objCords, objectsMap):
    r = center_circle_range
    for i in range(len(objCords)):
        if objCords[i] != 0:
            x = int((objCords[i]["x"] / objCords[i]["n"]))
            y = int((objCords[i]["y"] / objCords[i]["n"]))
            for j in range(-r, r + 1):
                for k in range(-r, r + 1):
                    if len(objectsMap) > x + j >= 0 \
                            and len(objectsMap[x]) > y + k >= 0 \
                            and abs(j) + abs(k) <= r:
                        objectsMap[x + j][y + k] = i


def count_bounding_box(idsMap, lev, objCords, objectsMap):
    for i in range(len(objectsMap)):
        for j in range(len(objectsMap[i])):
            objId = int(objectsMap[i][j])
            if idsMap[objId] < lev:
                objectsMap[i][j] = 0
                objCords[objId] = 0
            else:
                objCords[objId]["x"] += i
                objCords[objId]["y"] += j
                objCords[objId]["n"] += 1


def colorImg(img):
    objectsMap, maxObjId = findObjects(img)
    colorStep = 1 / maxObjId
    colorsMap = []
    unique_adder = random()
    for i in range(len(img)):
        colorsMap.append([])
        for j in range(len(img[i])):
            h = colorStep * objectsMap[i][j]
            if not h == 0:
                h += unique_adder
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


def fun():
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
        drawPlanesImage(i + 1, res)
    plt.tight_layout()  # Aby obrazy znajdowały się obok siebie
    plt.show()
    fig.savefig("planes.pdf")
    plt.close()


fun()
