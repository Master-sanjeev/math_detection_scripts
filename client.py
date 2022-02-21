import sys
import requests
import time



if len(sys.argv) < 2:
    print("Please specify the image")
    sys.exit(0)

f = open(sys.argv[1], "rb")
content = f.read()
# print(content)
f.close()

url = "http://0.0.0.0:5001/"
my_obj = {"file":content}

start = time.time()
print("Sending Request")
x = requests.post(url, content, headers={'Content-Type': 'application/octet-stream'})
print("Response received, Response time ", time.time()-start, " seconds")
# f = open("response.png", "wb")  
# f.write(x.content)
# f.close()
print(x.text)
# print(x.content)
