from sklearn.preprocessing import PolynomialFeatures
import datetime

def convertToFrequency(path, path2, outputPath, featIndexes, targetIndex):
    "Converts chosen features to their corresponding click frequency."
    #Could be done with arrays way faster 
    #but it's a one time operation and loops are clearer/smaller memory impact

    #Count all values for selected indexes
    training = rawGenerator(path)
    clickCount = [{} for i in range(max(featIndexes)+1)]
    count = [{} for i in range(max(featIndexes)+1)]
    rowIndex = 0
    for row in training:
        clicked = int(row[targetIndex])
        for i in featIndexes:
            featCounts = count[i]
            featClickCounts = clickCount[i]
            featValue = row[i]
            try: #only need to add once to dict << total number of appearances
                featCounts[featValue] += 1
                if clicked: 
                    featClickCounts[featValue] += 1
            except KeyError:
                featCounts[featValue] = 1
                if int(row[targetIndex]):
                    featClickCounts[featValue] = 1
                else: #Gotta initiate anyway
                    featClickCounts[featValue] = 0
        if rowIndex % 100000 == 0: print(rowIndex/420000, "%")
        rowIndex += 1

    for i in clickCount:
        print(len(i.keys()))
    input()
                    
    #Calculate associated frequencies
    frequency = [{} for i in range(max(featIndexes)+1)]
    for i in featIndexes:
        featCounts = count[i]
        featClickCounts = clickCount[i]
        for key in featCounts.keys():
            frequency[i][key] = featClickCounts[key] / featCounts[key]

    #Write in output file
    print("Writing results.")
    training = rawGenerator(path2)
    outputFile = open(outputPath, "w")
    outputFile.write("id,click,hour,C1,banner_pos,site_id,site_domain,site_category,app_id,app_domain,app_category,device_id,device_ip,device_model,device_type,device_conn_type,C14,C15,C16,C17,C18,C19,C20,C21\n")
    rowIndex = 0
    for row in training: 
        for i in featIndexes:
            try:
##                row[i-1] = str(frequency[i][row[i-1]]) #replace row features by their frequency
                row[i] = str(frequency[i][row[i]]) #replace row features by their frequency
            except KeyError:
##                row[i-1] = "0.5"
                row[i] = "0.5"
        rowString = ",".join(row) + "\n"
        outputFile.write(rowString)
        rowIndex += 1
        if rowIndex % 100000 == 0: print(rowIndex/42000, "%")
    outputFile.close()

def splitDateFeature(row):
    "Replaces the date feature with year,month,day,hour features (inplace)."
    date = row[2]
    temp0 = int(date/10000)
    temp1 = int(date/100)
    year = int(date / 1000000) 
    month = temp0 - year * 100
    day = temp1 - temp0 *100 
    hour = date - temp1 * 100 
    row.append(year)
    row.append(month)
    row.append(day)
    row.append(hour)

    
def addWorkingFeature(row):
    "Adds a feature representing whether the customer clicked during his worktime (inplace)."
    year, month, day, hour = row[-4:] #get time features
    #check if week-end
    date =  datetime.datetime(2000+year, month, day)
    weekday = date.weekday() #0:monday -> 6:sunday
    weekend = weekday == 5 or weekday==6
    #check if hour is during the day
    daytime = hour >= 9 and hour <= 17 

    working = int(daytime and not weekend)
    row.append(working)

def createPolynomialFeatures(row, PF):
    "Returns row with polynomial features included."
    newRow = PF.fit_transform(row) #2d array
    return newRow[0]
            
if __name__=="__main__":
    from avazuGenerator2 import rawGenerator, generatorWithFreq
##    PATH = "E:\\Users\\Etienne2\\Downloads\\train.csv"
##    PATH2 = "test.csv"
##    OUTPATH = "testfreq1.csv"
    PATH = "tenth.csv"
    PATH2 = "tenth.csv"
    OUTPATH = "tenthfreq1.csv"
    FEATINDEXES = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
    FEATINDEXES = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
    TARGETINDEX = 1

    convertToFrequency(PATH, PATH2, OUTPATH, FEATINDEXES, TARGETINDEX)
##    PATH = "tenthfreq.csv"
##    OUTPATH = "tenthfreq1.csv"
##    PATH = "hundredthfreq.csv"
##    OUTPATH = "hundredthfreq1.csv"
##    gen = rawGenerator(PATH)
##    f = open(OUTPATH,"w")
##    index = 0
##    for row in gen:
##        temp = [float(i) for i in row]
##        row = temp
##        splitDateFeature(row)
##        addWorkingFeature(row)
##        row = [str(i) for i in row]
##        f.write(",".join(row)+"\n")
##        index += 1
##        if index % 10000 == 0: print(index/42000)
##    f.close()
                
    
    
