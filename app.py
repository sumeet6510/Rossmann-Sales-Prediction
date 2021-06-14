import pickle

import numpy as np
from flask import Flask, request, render_template

loaded_model = pickle.load(open('Sales prediction', 'rb'))

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route("/submit", methods=['POST'])
def submit():
    if request.method == 'POST':
        Days_of_Week = request.form['Days of Week']
        Customers = request.form['Customers']
        store_open_close = request.form['subject']
        State_Holiday = request.form['State Holiday']
        School_Holiday = request.form['School Holiday']
        Store_Type = request.form['Store Type']
        Assortment_Type = request.form['Assortment Type']
        Competition_Distance = request.form['Competition Distance']
        Competition_month = request.form['Competition Month']
        Competition_year = request.form['Competition Year']
        Promo2 = request.form['Promo2']
        feature_list = [Days_of_Week, Customers, store_open_close, State_Holiday, School_Holiday,
                Store_Type, Assortment_Type, Competition_Distance, Competition_month, Competition_year, Promo2]
        array = np.array(feature_list)
        array=array.reshape(1,11)
        result = loaded_model.predict(array)
        return render_template('index.html', final_result = result)

if __name__ == '__main__':
    app.run()
