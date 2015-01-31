from avazuGenerators import rawGenerator

PATH = "test.csv"

OUTPATH1 = "testsetComp.csv"
OUTPATH2 = "testsetPhone.csv"
HEADERROW = 'id,hour,C1,banner_pos,site_id,site_domain,site_category,app_id,app_domain,app_category,device_id,device_ip,device_model,device_type,device_conn_type,C14,C15,C16,C17,C18,C19,C20,C21\n'

COMPINDEX = [4,7]
PHONEINDEX = [7,10]

gen = rawGenerator(PATH)

f0 = open(OUTPATH1,"w")
f1 = open(OUTPATH2,"w")

f0.write(HEADERROW)
f1.write(HEADERROW)

rowIndex = 0
for row in gen:
    app_id = row[7]
    if app_id == "ecad2386": #null value
        newRow = row[:7] + row[10:]
        curFile = f0
    else:
        newRow = row[:4] + row[7:]
        curFile = f1
    stringRow = ",".join(newRow) + "\n"
    curFile.write(stringRow)

    if rowIndex % 40000 == 0: print(rowIndex/40200,"%")
    rowIndex += 1

f0.close()
f1.close()
        



