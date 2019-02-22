import gdal
import numpy 
import math
import random
from pandas import Series,DataFrame
class Cluster:
    """Cluster in Gray"""
    def __init__(self, initCenter):
        self.center = initCenter
        self.pixelList = []

a = numpy.array([[1,2],[4,2],[2,3]])
df = DataFrame(a)
c= df.

print(b)

dataset = gdal.Open("before.img")
im_bands = dataset.RasterCount #波段数
imgX = dataset.RasterXSize #栅格矩阵的列数
imgY = dataset.RasterYSize #栅格矩阵的行数
imgArray = dataset.ReadAsArray(0,0,imgX,imgY)#获取数据
K = 3
# print(imgArray[9,0,9])
# print(type(imgArray))
clusterList = []
    # Generate K cluster centers randomly
for i in range(0, K):
    randomX = random.randint(0, imgX - 1)
    randomY = random.randint(0, imgY - 1)
    duplicated = False
    for cluster in clusterList:
        if (cluster.center[0] == imgArray[0,randomX, randomY] and
            cluster.center[1] == imgArray[1,randomX, randomY] and
            cluster.center[2] == imgArray[2,randomX, randomY] and
            cluster.center[3] == imgArray[3,randomX, randomY] and
            cluster.center[4] == imgArray[4,randomX, randomY] and
            cluster.center[5] == imgArray[5,randomX, randomY] and
            cluster.center[6] == imgArray[6,randomX, randomY] 
            ):
            duplicated = True
            break
    if not duplicated:
        clusterList.append(Cluster(numpy.array([imgArray[0,randomX, randomY],
                                                imgArray[1,randomX, randomY],
                                                imgArray[2,randomX, randomY],
                                                imgArray[3,randomX, randomY],
                                                imgArray[4,randomX, randomY],
                                                imgArray[5,randomX, randomY],
                                                imgArray[6,randomX, randomY]
                                                ],
                                                dtype = numpy.uint8)))
print(imgArray[:,0,0])
print(imgArray[:,0,0][0])
    # if not duplicated:
    #     clusterList.append(Cluster(numpy.array([imgArray[randomX, randomY, 0],
    #                                             imgArray[randomX, randomY, 1],
    #                                             imgArray[randomX, randomY, 2]],
    #                                             dtype = numpy.uint8)))
