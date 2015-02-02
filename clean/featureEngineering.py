



def crossSiteApp(example):
    cross_id = example[3]+"_"+example[6]
    cross_domain = example[4]+"_"+example[7]
    cross_category = example[5]+"_"+example[8]
    example.append(cross_id)
    example.append(cross_domain)
    example.append(cross_category)

def replaceHour(example):
    date = example.pop(0)
    day = date[4:6]
    hour = date[6:8]
    example.insert(0,day) #to keep the same indexes for the other features
    example.append(hour)
    

def createInteractionFunc(i0, i1):
    def addInteraction(example):
        newFeat = example[i0] + "_" + example[i1]
        example.append(newFeat)
    return addInteraction
