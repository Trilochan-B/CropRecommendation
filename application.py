from flask import Flask,render_template,request
from src.pipeline.predict_pipeline import predictPipeline
import numpy as np

application = Flask(__name__)
app = application
predictObj = predictPipeline()

@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        n = request.form['input1']
        p = request.form['input2']
        k = request.form['input3']
        t = request.form['input4']
        h = request.form['input5']
        ph = request.form['input6']
        r = request.form['input7']

        features = np.array([n,p,k,t,h,ph,r])
        features = features[np.newaxis,...]

        pred = predictObj.predict(features)
        return render_template("index.html", pred = pred)
    return render_template("index.html")
if __name__ == "__main__":
    app.run(debug = True)