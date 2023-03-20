from datetime import datetime
import cv2
import flask
from tkinter import *
import torch as torch
from PIL import Image

width = 150
height = 150

app = flask.Flask(__name__)
app.config['UPLOAD_FOLDER'] = "D:\\website"

app.config['MAX_CONTENT_PATH'] = 9999999
DATETIME_FORMAT = "%Y-%m-%d_%H-%M-%S-%f"

@app.route('/')
def home():
    return flask.render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    return flask.render_template('index.html', prediction_text='Percent with heart disease is {}'.format("????"))

@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if flask.request.method == 'POST' and flask.request.files['file']:
        f = flask.request.files['file']
        #Save to a file on a Flask server
        f.save("D:\\website\\temp.jpg")
        now_time = datetime.now().strftime(DATETIME_FORMAT)
        img_savename = f"D:\\website\\static\\{now_time}.jpg"
        #Read the temp file saved
        img = cv2.imread('D:\\website\\temp.jpg')
        #Change color to gray
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #Works better on small image size, hence, 320
        results = model(img, size=320)  # includes NMS
        #Add bounding box to the picture
        results.render()
        #Save picture with bounding boxes
        Image.fromarray(results.ims[0]).save(img_savename)
        filename = './static/' + now_time +'.jpg'
        #return flask.render_template('index.html', prediction_text='Картинка содержит логотип компании: {}'.format(results), img=filename)
        #Display a picture with bounding boxes
        return flask.render_template('index.html', img=filename)
    else:
        return flask.render_template('index.html')


if __name__ == "__main__":
    #Show a path to weights of your model here
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='D:\\Logo_YOLO\\yolov5\\runs\\train\\yolo_logo_detection\\weights\\best.pt',force_reload = True)
    model.eval()
    app.run()
