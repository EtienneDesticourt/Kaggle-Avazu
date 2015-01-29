from avazuGenerator2 import rawGenerator
from sklearn.preprocessing import PolynomialFeatures


def convertToFrequency(path, outputPath, featIndexes, targetIndex):
    "Converts chosen features to their corresponding click frequency."
    #Could be done with arrays way faster 
    #but it's a one time operation and loops are clearer/smaller memory impact

    #Count all values for selected indexes
    training = rawGenerator(path)
    clickCount = [{} for i in featIndexes]
    count = [{} for i in featIndexes]
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

                    
    #Calculate associated frequencies
    frequency = [{} for i in featIndexes]
    for i in featIndexes:
        featCounts = count[i]
        featClickCounts = clickCount[i]
        for key in featCounts.keys():
            frequency[i][key] = featClickCounts[key] / featCounts[key]

    #Write in output file
    training = rawGenerator(path)
    outputFile = open(outputPath, "w")
    for row in training: 
        for i in featIndexes:
            row[i] = frequency[i][row[i]] #replace row features by their frequency
        rowString = "".join(row) + "\n"
        outputFile.write(rowString)
    outputFile.close()

def splitDateFeature(row):
    "Replaces the date feature with year,month,day,hour features."
    date = row.pop(1) #remove date
    year = date[:2] ; row.append(year)
    month = date[2:4] ; row.append(month)
    day = date[4:6] ; row.append(day)
    hour = date[6:8] ; row.append(hour)

    
def addWorkingFeature(row):
    "Add a feature representing whether the customer clicked during his worktime."
    year, month, day, hour = row[-4:] #get time features
    #check if week-end
    date =  datetime.datetime(2000+int(year),int(month), int(day))
    weekday = date.weekday() #0:monday -> 6:sunday
    weekend = weekday == 5 or weekday==6
    #check if hour is during the day
    daytime = hour >= 9 or hour <= 17 

    working = daytime and not weekend
    row.append(working)

def createPolynomialFeatures(row):
    PF = PolynomialFeatures(interaction_only=True)
    newRow = PF.fit_transform(row) #2d array
    return newRow[0]
            
    
                    
                    
                
    
    
