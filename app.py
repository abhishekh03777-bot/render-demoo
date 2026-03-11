from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open('pipe.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    BattingTeam = request.form['BattingTeam']
    BowlingTeam = request.form['BowlingTeam']
    City = request.form['City']

    target = int(request.form['target'])
    score = int(request.form['score'])
    overs = float(request.form['overs'])
    wickets = int(request.form['wickets'])

    runs_left = target - score
    balls_left = 120 - int(overs * 6)
    wickets_left = 10 - wickets

    CRR = score / overs
    RRR = (runs_left * 6) / balls_left

    input_data = pd.DataFrame({
        'BattingTeam':[BattingTeam],
        'BowlingTeam':[BowlingTeam],
        'City':[City],
        'runs_left':[runs_left],
        'balls_left':[balls_left],
        'wickets_left':[wickets_left],
        'total_run_x':[target],
        'current_score':[score],
        'CRR':[CRR],
        'RRR':[RRR]
    })

    result = model.predict_proba(input_data)

    win = round(result[0][1]*100)
    loss = round(result[0][0]*100)

    return render_template('index.html',
                           prediction_text=f"{BattingTeam} Win Probability: {win}%",
                           loss_text=f"{BowlingTeam} Win Probability: {loss}%")


if __name__ == "__main__":
    app.run(debug=True)