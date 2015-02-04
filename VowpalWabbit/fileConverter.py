


def csvToVW(pathCSV, pathVW):
    csv = open(pathCSV, "r")
    VW = open(pathVW,"w")

    headerRow = csv.readline().split("\n")
    
    #Get index of label and ID and remove them
    labelIndex = headerRow.index("click")
    idIndex = headerRow.index("ID")
    headerRow.pop(labelIndex)
    headerRow.pop(idIndex)   

    
    rowIndex = 0
    row = headerRow 
    while row != "":       
        row = csv.readline().split("\n") #Read csv row
        #Create vowpal row
        formattedRow = ""
        #Add label
        label = int(row[labelIndex]) or -1 #We take 1 and -1 as labels
        formattedRow += label + "|"
        #Remove ID
        row.pop(idIndex)    
        #Add features
        features = ""
        for featIndex in range(len(row)):
            features += headerRow[featIndex] + ":" + row[featIndex]
        formattedRow += features


        
        rowIndex += 1
    
