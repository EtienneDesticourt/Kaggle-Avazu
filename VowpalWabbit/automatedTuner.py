import subprocess
import time
"vw train.vw -f avazu.model.vw --loss_function logistic -b26  --l1 0.000000003375  -l 0.05 -c -k --passes 10  --holdout_after 32377422"

baseArgs = {"-f":"avazu.model.vw",
            "--loss_function":"logistic",
            "-b":"22",
            "--l1":"000000003375",
            "-l":"0.05",
            "--holdout_after":"32377422",
            "-c":"",
            "-k":"",
            "--passes":"10"}

def runModel(path, args):
    line = "vw " + path
    for arg in args:
        line  += " " + arg + " " + args[arg]
    print(line)
    output = subprocess.check_output(line, shell=True, universal_newlines=True)
    print(output)
    return output
    
    
def parseOutput(text):
    lines = text.split("\n")
    for line in lines:
        if "average loss" in line:
            loss = line.split(" = ")[1]
            print(loss)
            return loss
    assert "Should never reach this point."

def logResults(path, args, loss, trainingTime):
    output = loss
    for arg in args:
        print(arg)
        output += "," + arg
    output += "," + str(trainingTime/60) + "\n"
    f = open(path,"a")
    f.write(output)
    f.close()

def executeAttempt(trainPath, scoresPath, args):
    start = time.time()
    output = runModel(trainPath, args)
    print(output)
    score = parseOutput(output)
    end = time.time()
    logResults(scoresPath, args, score, end-start)
    

TRAINPATH = "temp.vw"
SCOREPATH = "scores.csv"

baseArgs["--passes"] = "1"
baseArgs["--holdout_after"] = "1000"
baseArgs["--l1"] = "0"
baseArgs["-l"] = "10"
executeAttempt(TRAINPATH, SCOREPATH, baseArgs)

##baseArgs[
##executeAttempt(TRAINPATH, SCOREPATH, baseArgs)







