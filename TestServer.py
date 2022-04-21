from flask import Flask, request
import os

app = Flask(__name__)
@app.route('/', methods=(["POST"]))
def hello():
    if request.method == "POST":
        try:
          os.mkdir("pdf")
        except:
          pass
        f = open("pdf/new_file.pdf", "wb")
        f.write(request.data)
        print(request.data)
        print(type(request.data))
        f.close()
    
    return None



if __name__ == "__main__":
    app.run(host ='0.0.0.0', port = 5001,debug = True)
