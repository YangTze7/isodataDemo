import gdal
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
    argvK = 3
    argvTN = 2
    argvTS = 1
    argvTC = 4
    argvL = 1
    argvI = 4
    # fileName = "before.img"
    
    

    if isRGB:

        dataset = gdal.Open("before.img")
        
   
        # image = Image.open(inputFilename)
        result = ISODATAKit.doISODATARGB(dataset, argvK, argvTN, argvTS, argvTC, argvL, argvI)
        # plot.figure("Result")
        # plot.imshow(result,cmap="lines")
        # result.save("result.jpg")
        # plot.show()
    else:
        image = Image.open(inputFilename).convert("L")
        result = ISODATAKit.doISODATAGray(image, argvK, argvTN, argvTS, int(argvTC), argvL, argvI)
        plot.figure("Result")
        plot.imshow(result, cmap = "gray")
        
        plot.show()

    result.save(outputFilename)


if __name__ == '__main__':
    main()


