


def csvToVW(pathCSV, pathVW, v=False):
    csv = open(pathCSV, "r")
    VW = open(pathVW,"w")

    headerRow = csv.readline()[:-1].split(",")
    
    #Get index of label and ID and remove them
#    labelIndex = headerRow.index("click")
    idIndex = headerRow.index("id")
#    headerRow.pop(labelIndex)
    headerRow.pop(idIndex)   

    
    rowIndex = 0
    row = headerRow
    rows = []
    dicid = {}
    dicip = {}
    while 1:       
        row = csv.readline()[:-1].split(",") #Read csv row
        if len(row) == 1: break #split "" by "," -> []
        #Create vowpal row
        formattedRow = ""
        #Add label
        label = 1# int(row[labelIndex]) or -1 #We take 1 and -1 as labels
        ID = row[idIndex]
        formattedRow += str(label) + " '" + ID + " |"
        #Remove ID
        #row.pop(labelIndex)
        row.pop(idIndex)
        #Add features
        features = "" #namespace
        ID = row[9]
        IP = row[10]
        if ID in dicid:
            dicid[ID] += 1
        else:
            dicid[ID] = 1
        if IP in dicip:
            dicip[IP] += 1
        else:
            dicip[IP] = 1
            
        for featIndex in range(len(row)):
            features += chr(65+featIndex) + " " + headerRow[featIndex] + "=" + row[featIndex]+" |"
        features += " |iA A:"+str(min(dicid[ID],8))+" |iB B:"+str(min(8,dicip[IP]))
        formattedRow += features

        rows.append(formattedRow)

        if len(rows) >= 1000000:
            VW.write("\n".join(rows)+"\n")
            rows = []
        
        rowIndex += 1
        if rowIndex % 1000000 == 0 and v:
            print(rowIndex)
    
    VW.write("\n".join(rows)+"\n")
    VW.close()
    csv.close()

pathIn = "..\\data\\train.csv"
pathOut = "trainSep2.vw"
##>>> pathIn = "trainSep.vw"
##>>> pathOut ="trainSep2.vw"
##>>> f = open(pathIn,"r")
##>>> f2= open(pathOut, "w")
##>>> while 1:
##	r = f.readline()
##	if r == "":break
##	f2.write(r[:-3]+"\n")
