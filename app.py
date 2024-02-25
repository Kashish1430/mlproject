from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from src.pipelines.predict_pipeline import CustomData, PredictPipleline
from sklearn.preprocessing import StandardScaler
import numpy as np

application = Flask(__name__)

app = application

## ROute for the home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data = CustomData(
            year = int(request.form.get('year')),
            month = int(request.form.get('month')),
            toxicity = float(request.form.get('toxicity')),
            Fulldate = request.form.get('Fulldate'),
            label = request.form.get('label')
            )
        df = data.get_data_as_data_frame()
        pred_pipe = PredictPipleline()
        results = pred_pipe.predict(df)
        return render_template('home.html', results=np.round(results[0],2))
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
    