#!/usr/bin/env python3
# coding: utf-8




import numpy
import math
import random
import gdal
import cv2

# Class
class Pixel:
    """Pixel"""
    def __init__(self, initX: int, initY: int, initColor):
        self.x = initX
        self.y = initY
        self.color = initColor

class Cluster:
    """Cluster in Gray"""
    def __init__(self, initCenter):
        self.center = initCenter
        self.pixelList = []

class ClusterPair:
    """Cluster Pair"""
    def __init__(self, initClusterAIndex: int, initClusterBIndex: int, initDistance):
        self.clusterAIndex = initClusterAIndex
        self.clusterBIndex = initClusterBIndex
        self.distance = initDistance

# Functions

# Gray


# RGB
def distanceBetween(colorA, colorB) -> float:
    aveR = float(int(colorA[0]) + int(colorB[0])) / 2
    dR = int(colorA[0]) - int(colorB[0])
    dG = int(colorA[1]) - int(colorB[1])
    dB = int(colorA[2]) - int(colorB[2])
    d4 = int(colorA[3]) - int(colorB[3])
    d5 = int(colorA[4]) - int(colorB[4])
    d6 = int(colorA[5]) - int(colorB[5])
    d7 = int(colorA[6]) - int(colorB[6])
    return math.sqrt((dR**2)+(dG**2)+(dB**2)+(d4**2)+(d5**2)+(d6**2)+(d7**2))
    # return math.sqrt((2 + aveR / 256) * (dR ** 2) + 4 * (dG ** 2) + (2 + (255 - aveR) / 256) * (dB ** 2))

def gray2rgb(val):
   

    #red
    if (val<128):
        r = 0 
    elif(val<192):
        r = 255/64*(val-128)
    else:  
        r=255 


    #green
    if (val<64):
        g = 255/64*val 
    elif(val<192):
        g = 255 
    else:
        g= -255/64*(val - 192)+255 
    #blue

    if (val<64):
        b = 255 
    elif(val<128):
        b = -255/64*(val - 128) 
    else:
        b=0 
    rgb = []
    rgb.append(r)
    rgb.append(g)
    rgb.append(b)
    return rgb 


def doISODATARGB(dataset, K: int, TN: int, TS: float, TC:int, L: int, I: int):

   
    im_bands = dataset.RasterCount #波段数
    imgX = dataset.RasterXSize #栅格矩阵的列数
    imgY = dataset.RasterYSize #栅格矩阵的行数
    im_geotrans = dataset.GetGeoTransform()  #仿射矩阵
    im_proj = dataset.GetProjection() #地图投影信息
    imgArray = dataset.ReadAsArray(0,0,imgX,imgY)#获取数据
    
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

    # Iteration
    iterationCount = 0
    didAnythingInLastIteration = True
    while True:
        iterationCount += 1

        # Clear the pixel lists of all clusters
        for cluster in clusterList:
            cluster.pixelList.clear()
        print("------")
        print("Iteration: {0}".format(iterationCount))

        # Classify all pixels into clusters
        print("Classifying...", end = '', flush = True)
        for row in range(0, imgX):
            for col in range(0, imgY):
                targetClusterIndex = 0
                targetClusterDistance = distanceBetween(imgArray[:,row, col], clusterList[0].center)
                # Classify
                for i in range(1, len(clusterList)):
                    currentDistance = distanceBetween(imgArray[:,row, col], clusterList[i].center)
                    if currentDistance < targetClusterDistance:
                        targetClusterDistance = currentDistance
                        targetClusterIndex = i
                clusterList[targetClusterIndex].pixelList.append(Pixel(row, col, imgArray[:,row, col]))
        print(" Finished.")

        # Check TN
        gotoNextIteration = False
        for i in range(len(clusterList) - 1, -1, -1):
            if len(clusterList[i].pixelList) < TN:
                # Re-classify
                clusterList.pop(i)
                gotoNextIteration = True
                break
        if gotoNextIteration:
            print("TN checking not passed.")
            continue
        print("TN checking passed.")

        # Recalculate the centers
        print("Recalculating the centers...", end = '', flush = True)
        for cluster in clusterList:
            sumR = 0.0
            sumG = 0.0
            sumB = 0.0
            sum4 = 0.0
            sum5 = 0.0
            sum6 = 0.0
            sum7 = 0.0
           
            for pixel in cluster.pixelList:
                sumR += int(pixel.color[0])
                sumG += int(pixel.color[1])
                sumB += int(pixel.color[2])
                sum4 += int(pixel.color[3])
                sum5 += int(pixel.color[4])
                sum6 += int(pixel.color[5])
                sum7 += int(pixel.color[6])
            aveR = round(sumR / len(cluster.pixelList))
            aveG = round(sumG / len(cluster.pixelList))
            aveB = round(sumB / len(cluster.pixelList))
            ave4 = round(sum4 / len(cluster.pixelList))
            ave5 = round(sum5 / len(cluster.pixelList))
            ave6 = round(sum6 / len(cluster.pixelList))
            ave7 = round(sum7 / len(cluster.pixelList))
            
            if (aveR != cluster.center[0] and
                aveG != cluster.center[1] and
                aveB != cluster.center[2] and
                ave4 != cluster.center[3] and
                ave5 != cluster.center[4] and
                ave6 != cluster.center[5] and
                ave7 != cluster.center[6] 
                ):
                didAnythingInLastIteration = True
            cluster.center = numpy.array([aveR, aveG, aveB, ave4,ave5,ave6,ave6,ave7], dtype = numpy.uint8)
        print(" Finished.")
        if iterationCount > I:
            break
        if not didAnythingInLastIteration:
            print("More iteration is not necessary.")
            break

        # Calculate the average distance
        print("Preparing for Merging and Spliting...", end = '', flush = True)
        aveDisctanceList = []
        sumDistanceAll = 0.0
        for cluster in clusterList:
            currentSumDistance = 0.0
            for pixel in cluster.pixelList:
                currentSumDistance += distanceBetween(pixel.color, cluster.center)
            aveDisctanceList.append(float(currentSumDistance) / len(cluster.pixelList))
            sumDistanceAll += currentSumDistance
        aveDistanceAll = float(sumDistanceAll) / (imgX * imgY)
        print(" Finished.")

        if (len(clusterList) <= K / 2) or not (iterationCount % 2 == 0 or len(clusterList) >= K * 2):
            # Split
            print("Split:", end = '', flush = True)
            beforeCount = len(clusterList)
            for i in range(len(clusterList) - 1, -1, -1):
                currentSD = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                for pixel in clusterList[i].pixelList:
                    currentSD[0] += (int(pixel.color[0]) - int(clusterList[i].center[0])) ** 2
                    currentSD[1] += (int(pixel.color[1]) - int(clusterList[i].center[1])) ** 2
                    currentSD[2] += (int(pixel.color[2]) - int(clusterList[i].center[2])) ** 2
                    currentSD[3] += (int(pixel.color[3]) - int(clusterList[i].center[3])) ** 2
                    currentSD[4] += (int(pixel.color[4]) - int(clusterList[i].center[4])) ** 2
                    currentSD[5] += (int(pixel.color[5]) - int(clusterList[i].center[5])) ** 2
                    currentSD[6] += (int(pixel.color[6]) - int(clusterList[i].center[6])) ** 2
                currentSD[0] = math.sqrt(currentSD[0] / len(clusterList[i].pixelList))
                currentSD[1] = math.sqrt(currentSD[1] / len(clusterList[i].pixelList))
                currentSD[2] = math.sqrt(currentSD[2] / len(clusterList[i].pixelList))
                currentSD[3] = math.sqrt(currentSD[3] / len(clusterList[i].pixelList))
                currentSD[4] = math.sqrt(currentSD[4] / len(clusterList[i].pixelList))
                currentSD[5] = math.sqrt(currentSD[5] / len(clusterList[i].pixelList))
                currentSD[6] = math.sqrt(currentSD[6] / len(clusterList[i].pixelList))
               
                # Find the max in SD of R, G and B
                maxSD = currentSD[0]
                for j in (1, 2):
                    maxSD = currentSD[j] if currentSD[j] > maxSD else maxSD
                if (maxSD > TS) and ((aveDisctanceList[i] > aveDistanceAll and len(clusterList[i].pixelList) > 2 * (TN + 1)) or (len(clusterList) < K / 2)):
                    gamma = 0.5 * maxSD
                    clusterList[i].center[0] += gamma
                    clusterList[i].center[1] += gamma
                    clusterList[i].center[2] += gamma
                    clusterList[i].center[3] += gamma
                    clusterList[i].center[4] += gamma
                    clusterList[i].center[5] += gamma
                    clusterList[i].center[6] += gamma
                  
                    clusterList.append(Cluster(numpy.array([clusterList[i].center[0],
                                                            clusterList[i].center[1],
                                                            clusterList[i].center[2],
                                                            clusterList[i].center[3],
                                                            clusterList[i].center[4],
                                                            clusterList[i].center[5],
                                                            clusterList[i].center[6],
                                                            ],
                                                            dtype = numpy.uint8)))
                    clusterList[i].center[0] -= gamma * 2
                    clusterList[i].center[1] -= gamma * 2
                    clusterList[i].center[2] -= gamma * 2
                    clusterList[i].center[3] -= gamma * 2
                    clusterList[i].center[4] -= gamma * 2
                    clusterList[i].center[5] -= gamma * 2
                    clusterList[i].center[6] -= gamma * 2
                    clusterList.append(Cluster(numpy.array([clusterList[i].center[0],
                                                            clusterList[i].center[1],
                                                            clusterList[i].center[2],
                                                            clusterList[i].center[3],
                                                            clusterList[i].center[4],
                                                            clusterList[i].center[5],
                                                            clusterList[i].center[6]
                                                            ],
                                                            dtype = numpy.uint8)))
                    clusterList.pop(i)
            print(" {0} -> {1}".format(beforeCount, len(clusterList)))
        elif (iterationCount % 2 == 0) or (len(clusterList) >= K * 2) or (iterationCount == I):
            # Merge
            print("Merge:", end = '', flush = True)
            beforeCount = len(clusterList)
            didAnythingInLastIteration = False
            clusterPairList = []
            for i in range(0, len(clusterList)):
                for j in range(0, i):
                    currentDistance = distanceBetween(clusterList[i].center, clusterList[j].center)
                    if currentDistance < TC:
                        clusterPairList.append(ClusterPair(i, j, currentDistance))

            clusterPairListSorted = sorted(clusterPairList, key = lambda clusterPair: clusterPair.distance)
            newClusterCenterList = []
            mergedClusterIndexList = []
            mergedPairCount = 0
            for clusterPair in clusterPairList:
                hasBeenMerged = False
                for index in mergedClusterIndexList:
                    if clusterPair.clusterAIndex == index or clusterPair.clusterBIndex == index:
                        hasBeenMerged = True
                        break
                if hasBeenMerged:
                    continue
                newCenterR = int((len(clusterList[clusterPair.clusterAIndex].pixelList) * float(clusterList[clusterPair.clusterAIndex].center[0]) + len(clusterList[clusterPair.clusterBIndex].pixelList) * float(clusterList[clusterPair.clusterBIndex].center[0])) / (len(clusterList[clusterPair.clusterAIndex].pixelList) + len(clusterList[clusterPair.clusterBIndex].pixelList)))
                newCenterG = int((len(clusterList[clusterPair.clusterAIndex].pixelList) * float(clusterList[clusterPair.clusterAIndex].center[1]) + len(clusterList[clusterPair.clusterBIndex].pixelList) * float(clusterList[clusterPair.clusterBIndex].center[1])) / (len(clusterList[clusterPair.clusterAIndex].pixelList) + len(clusterList[clusterPair.clusterBIndex].pixelList)))
                newCenterB = int((len(clusterList[clusterPair.clusterAIndex].pixelList) * float(clusterList[clusterPair.clusterAIndex].center[2]) + len(clusterList[clusterPair.clusterBIndex].pixelList) * float(clusterList[clusterPair.clusterBIndex].center[2])) / (len(clusterList[clusterPair.clusterAIndex].pixelList) + len(clusterList[clusterPair.clusterBIndex].pixelList)))
                newCenter4 = int((len(clusterList[clusterPair.clusterAIndex].pixelList) * float(clusterList[clusterPair.clusterAIndex].center[3]) + len(clusterList[clusterPair.clusterBIndex].pixelList) * float(clusterList[clusterPair.clusterBIndex].center[3])) / (len(clusterList[clusterPair.clusterAIndex].pixelList) + len(clusterList[clusterPair.clusterBIndex].pixelList)))
                newCenter5 = int((len(clusterList[clusterPair.clusterAIndex].pixelList) * float(clusterList[clusterPair.clusterAIndex].center[4]) + len(clusterList[clusterPair.clusterBIndex].pixelList) * float(clusterList[clusterPair.clusterBIndex].center[4])) / (len(clusterList[clusterPair.clusterAIndex].pixelList) + len(clusterList[clusterPair.clusterBIndex].pixelList)))
                newCenter6 = int((len(clusterList[clusterPair.clusterAIndex].pixelList) * float(clusterList[clusterPair.clusterAIndex].center[5]) + len(clusterList[clusterPair.clusterBIndex].pixelList) * float(clusterList[clusterPair.clusterBIndex].center[5])) / (len(clusterList[clusterPair.clusterAIndex].pixelList) + len(clusterList[clusterPair.clusterBIndex].pixelList)))
                newCenter7 = int((len(clusterList[clusterPair.clusterAIndex].pixelList) * float(clusterList[clusterPair.clusterAIndex].center[6]) + len(clusterList[clusterPair.clusterBIndex].pixelList) * float(clusterList[clusterPair.clusterBIndex].center[6])) / (len(clusterList[clusterPair.clusterAIndex].pixelList) + len(clusterList[clusterPair.clusterBIndex].pixelList)))
                
                newClusterCenterList.append([newCenterR, newCenterG, newCenterB,newCenter4, newCenter5, newCenter6,newCenter7])
                mergedClusterIndexList.append(clusterPair.clusterAIndex)
                mergedClusterIndexList.append(clusterPair.clusterBIndex)
                mergedPairCount += 1
                if mergedPairCount > L:
                    break
            if len(mergedClusterIndexList) > 0:
                didAnythingInLastIteration = True
            mergedClusterIndexListSorted = sorted(mergedClusterIndexList, key = lambda clusterIndex: clusterIndex, reverse = True)
            for index in mergedClusterIndexListSorted:
                clusterList.pop(index)
            for center in newClusterCenterList:
                clusterList.append(Cluster(numpy.array([center[0], center[1], center[2],center[3], center[4], center[5],center[6]], dtype = numpy.uint8)))
            print(" {0} -> {1}".format(beforeCount, len(clusterList)))

    # Generate the new image martrix
    print("Over")
    print("Classified to {0} kinds.".format(len(clusterList)))
    newImgArray = numpy.zeros((7,imgX, imgY), dtype = numpy.uint8)
    for cluster in clusterList:
        for pixel in cluster.pixelList:
            newImgArray[0,pixel.x, pixel.y] = int(cluster.center[0])
            newImgArray[1,pixel.x, pixel.y] = int(cluster.center[1])
            newImgArray[2,pixel.x, pixel.y] = int(cluster.center[2])
            newImgArray[3,pixel.x, pixel.y] = int(cluster.center[3])
            newImgArray[4,pixel.x, pixel.y] = int(cluster.center[4])
            newImgArray[5,pixel.x, pixel.y] = int(cluster.center[5])
            newImgArray[6,pixel.x, pixel.y] = int(cluster.center[6])

    a2 = numpy.ones((3,imgX,imgY), dtype=numpy.uint8)  
    # for i in range(imgY):
    #     for j in range(imgX):
    #         a2[0,i,j] =(int)(newImgArray[0,i,j]-64)/42*255

    unic = numpy.unique(newImgArray[0])
    color = []
    for i in range(len(unic)): 
        color.append([random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)])


    for i in range(imgY):
        for j in range(imgX):
            for k in range(len(unic)):
                if(newImgArray[0,i,j] == unic[k]):
                    a2[0,i,j] = color[k][0]
                    a2[1,i,j] = color[k][1]
                    a2[2,i,j] = color[k][2]
    # for i in range(imgY):
    #     for j in range(imgX):
            # a2[0,i,j] = gray2rgb(newImgArray[0,i,j])[0]
            # a2[1,i,j] = gray2rgb(newImgArray[0,i,j])[1]
            # a2[2,i,j] = gray2rgb(newImgArray[0,i,j])[2]

            # if(newImgArray[0,i,j] in unic):
            
            #     a2[0,i,j] =
            #     a2[1,i,j] =newImgArray[0,i,j]*1/2
            #     a2[2,i,j] =newImgArray[0,i,j]
            
            # elif(newImgArray[0,i,j]>100):
            
            #     a2[0,i,j] =newImgArray[0,i,j]
            #     a2[1,i,j] =255
            #     a2[2,i,j] =newImgArray[0,i,j]
            
            # elif(newImgArray[0,i,j]>0):
            
            #     a2[0,i,j] =newImgArray[0,i,j]
            #     a2[1,i,j] =newImgArray[0,i,j]
            #     a2[2,i,j] =128
            
        

    driver = gdal.GetDriverByName("GTiff")
    IsoData = driver.Create("out31.tif", imgX, imgY, 3, gdal.GDT_Byte)
    # for i in range(3):
    #     IsoData.GetRasterBand(i+1).WriteArray(newImgArray[i])
    IsoData.SetGeoTransform(im_geotrans)              #写入仿射变换参数
    IsoData.SetProjection(im_proj)   #写入投影
    for i in range(3):                 
        IsoData.GetRasterBand(i+1).WriteArray(a2[i])
    # IsoData.GetRasterBand(1).WriteArray(a2)
    del dataset
    print("ISODATA SUCCESS")
    # im = cv2.imread('out31.tif', cv2.IMREAD_GRAYSCALE)
    # imC = cv2.applyColorMap(im, cv2.COLORMAP_JET)
    # cv2.imshow("result",imC)
   
    # return Image.fromarray(newImgArray, mode = "RGB")

if __name__ == '__main__':
    print("ERROR: Pleas don not run the module directly.")
