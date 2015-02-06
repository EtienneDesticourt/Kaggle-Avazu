


def csvToVW(pathCSV, pathVW, v=False):
    csv = open(pathCSV, "r")
    VW = open(pathVW,"w")

    headerRow = csv.readline()[:-1].split(",")
    
    #Get index of label and ID and remove them
    labelIndex = headerRow.index("click")
    idIndex = headerRow.index("id")
    headerRow.pop(labelIndex)
    headerRow.pop(idIndex)   

    
    rowIndex = 0
    row = headerRow 
    while 1:       
        row = csv.readline()[:-1].split(",") #Read csv row
        if len(row) == 0: break #split "" by "," -> []
        #Create vowpal row
        formattedRow = ""
        #Add label
        label = int(row[labelIndex]) or -1 #We take 1 and -1 as labels
        formattedRow += str(label) + "|"
        #Remove ID
        row.pop(labelIndex)
        row.pop(idIndex)
        #Add features
        features = "categorical " #namespace
        for featIndex in range(len(row)):
            features += headerRow[featIndex] + "=" + row[featIndex]+" "
        formattedRow += features

        VW.write(formattedRow+"\n")
        
        rowIndex += 1
        if rowIndex % 1000000 == 0 and v:
            print(rowIndex)
    VW.close()
    csv.close()
    
