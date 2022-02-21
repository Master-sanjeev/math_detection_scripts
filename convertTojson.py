import sys
import os

path = sys.argv[1]


txtfiles = []

for root, _, files in os.walk(path):
    for file in files:
        if file[-3:] == 'txt':
            txtfiles.append(root+"/"+file)




# path = file.split('.')[0]

for file in txtfiles :
    fileprefix = file.split('/')[-1].split('.')[0]
    # print(fileprefix)
    # coord = []
    jsonpath = file[0:-3] + 'json'
    f = open(file, 'r')

    data = "["
    lines = f.readlines()
    # print(lines)
    for line in lines:
        linedata = line.strip().split(" ")
        # print(linedata)
        data += fileprefix+','+linedata[1]+','+linedata[2]+','+linedata[3]+','+linedata[4]+','
    
    data = data[0:-1] + ']'
    if (data == ']'):
        data = '[]'
    print(data)
    f2 = open(jsonpath, 'w')
    f2.write(data)

    f.close()
    f2.close()
