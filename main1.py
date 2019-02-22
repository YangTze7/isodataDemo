import gdal
import ISO
def main():
    # Process the command line parameters
    # try:
    #     opts, args = getopt.getopt(sys.argv[1:],
    #         "h",
    #         ["help", "input=", "output=",
    #          "rgb", "gray",
    #          "K=", "TN=", "TS=", "TC=", "L=", "I="])
    # except getopt.GetoptError as error:
    #     print("Error: {error}".format(error = error))
    #     print(optionsHelp)
    #     exit()

    # print(__doc__)
    # if len(opts) == 0:
    #     print("Command line options are required.")
    #     print(optionsHelp)
    #     exit()

    # inputFilename = "test.tif"
    inputFilename = "before.img"
    outputFilename = "out.jpg"
    isRGB = True
    argvK = 8#类别数（期望）
    argvTN = 20#每个类别中样本最小数目
    argvTS = 1#每个类别的标准差
    argvTC = 0.5#每个类别间的最小距离
    argvL = 5#每次允许合并的最大类别对的数量
    argvI = 10#迭代次数
    # fileName = "before.img"
    
    

    if isRGB:

        dataset = gdal.Open("before.img")
        
   

        ISO.doISODATARGB(dataset, argvK, argvTN, argvTS, argvTC, argvL, argvI)


if __name__ == '__main__':
    main()


