


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
    rows = []
    while 1:       
        row = csv.readline()[:-1].split(",") #Read csv row
        if len(row) == 0: break #split "" by "," -> []
        #Create vowpal row
        formattedRow = ""
        #Add label
        label = int(row[labelIndex]) or -1 #We take 1 and -1 as labels
        ID = row[idIndex]
        formattedRow += str(label) + "'" + ID + " |"
        #Remove ID
        row.pop(labelIndex)
        row.pop(idIndex)
        #Add features
        features = "" #namespace
        for featIndex in range(len(row)):
            features += chr(65+featIndex) + " " + headerRow[featIndex] + "=" + row[featIndex]+" |"
        formattedRow += features

        rows.append(formattedRow[:-2])

        if len(rows) >= 1000000:
            VW.write("\n".join(rows)+"\n")
            rows = []
        
        rowIndex += 1
        if rowIndex % 1000000 == 0 and v:
            print(rowIndex)
    VW.close()
    csv.close()
    
##>>> pathIn = "trainSep.vw"
##>>> pathOut ="trainSep2.vw"
##>>> f = open(pathIn,"r")
##>>> f2= open(pathOut, "w")
##>>> while 1:
##	r = f.readline()
##	if r == "":break
##	f2.write(r[:-3]+"\n")
