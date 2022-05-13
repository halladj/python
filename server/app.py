import numpy as np
from flask import Flask, request, jsonify
import pickle
import pandas as pd
import json


app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))


@app.route('/api',methods=['POST'])
def predict():
    print(request)
    data = request.get_json(force=True)
    prediction = model.predict([[np.array(data['exp'])]])
    output = prediction[0]
    return jsonify(output)


def ValuePredictor(to_predict):
    price= model.predict(to_predict)
    return  json.dumps(price[0])

def createDataFram(data):
    formated_data= {
            "anti_glare": [data["antiGlare"]],
            "brand": [data["brand"]],
            "cpu_brand": [data["cpu"]["brand"]],
            "cpu_family": [data["cpu"]["family"]],
            "cpu_frequency": [data["cpu"]["frequency"]],
            "cpu_generation": [data["cpu"]["generation"]],
            "cpu_modifier": [data["cpu"]["modifier"]],
            "cpu_number_identifier": [data["cpu"]["numberIdentifier"]],
            "gpu_brand": [data["gpu"]["brand"]],
            "gpu_frequency": [data["gpu"]["frequency"]],
            "gpu_number_identifier": [data["gpu"]["numberIdentifier"]],
            "gpu_vram": [data["gpu"]["vram"]],
            "gpu_words_identifier": [data["gpu"]["wordsIdentifier"]],
            "hdd": [data["hdd"]],
            "ram": [data["ram"]],
            "ram_frequency": [data["ramFrequency"]],
            "ram_type": [data["ramType"]],
            "screen_refresh_rate": [data["screenRefreshRate"]],
            "screen_resolution": [data["screenResolution"]],
            "screen_size": [data["screenSize"]],
            "ssd": [data["ssd"]],
            "state": [data["state"]],
            "touch_screen": [data["touchScreen"]],

    }
    dataFrame=pd.DataFrame(formated_data)
    return dataFrame


def preprocessing(dataFrame):

    dataFrame["state"].replace([1, 0], [True, False], inplace=True)
    dataFrame["touch_screen"].replace([1, 0], [True, False], inplace=True)
    dataFrame["anti_glare"].replace([1, 0], [True, False], inplace=True)

    category_cols = {item: 'category' for item in ['ram_type', 'brand', 'cpu_family', "cpu_generation"
                                               , 'cpu_modifier',  'gpu_brand', 'gpu_number_identifier',
                                               'gpu_words_identifier','screen_resolution', 'cpu_brand', ]}

    dataFrame[['ram_type', 'brand', 'cpu_family',
        "cpu_generation",'cpu_modifier', 'gpu_brand',
        'gpu_number_identifier', 'gpu_words_identifier',
        'screen_resolution', 'cpu_brand']]= dataFrame[['ram_type', 'brand', 'cpu_family',
                                                       "cpu_generation",'cpu_modifier', 'gpu_brand',
                                                       'gpu_number_identifier', 'gpu_words_identifier',
                                                       'screen_resolution', 'cpu_brand']].astype('category')
    dataFrame.to_csv (r'json1.csv', index = None)
    df= pd.read_csv(r'json1.csv', dtype= category_cols)
    return df 
 
@app.route('/result', methods = ['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.get_json()
        dataFrame= createDataFram(to_predict_list)
        preparedData= preprocessing(dataFrame)
        result = ValuePredictor(preparedData)       
        if result != 0:
            return jsonify({"price":result})
        else:
            return "train go boom"

if __name__ == '__main__':
    app.run(port=3000, debug=True, host="192.168.1.18")

