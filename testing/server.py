from flask import Flask

app = Flask(__name__)

HOST = 'yaolaoban.eva0.nics.cc'
PORT = 5000

@app.route('/')
def hello():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run(host=HOST, port=PORT)
