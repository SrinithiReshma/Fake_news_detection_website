from flask import Flask, render_template, request
import pickle

# Load vectorizer and models
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('logistic_regression_model.pkl', 'rb') as f:
    LR = pickle.load(f)

with open('decision_tree_model.pkl', 'rb') as f:
    DT = pickle.load(f)

with open('random_forest_model.pkl', 'rb') as f:
    RF = pickle.load(f)

with open('gradient_boosting_model.pkl', 'rb') as f:
    GBC = pickle.load(f)

# Flask app setup
app = Flask(__name__)

def predict_news(news):
    # Preprocess the input and transform using vectorizer
    new_xv_test = vectorizer.transform([news])

    # Model predictions with probabilities
    pred_LR_prob = LR.predict_proba(new_xv_test)[:, 1]
    pred_DT_prob = DT.predict_proba(new_xv_test)[:, 1]
    pred_RF_prob = RF.predict_proba(new_xv_test)[:, 1]
    pred_GBC_prob = GBC.predict_proba(new_xv_test)[:, 1]

    # Weighted average probabilities
    rf_weight, lr_weight, dt_weight, gbc_weight = 0.9, 0.05, 0.1, 0.1
    weighted_prob = (
        rf_weight * pred_RF_prob +
        lr_weight * pred_LR_prob +
        dt_weight * pred_DT_prob +
        gbc_weight * pred_GBC_prob
    )

    # Final result
    real_percentage = weighted_prob[0] * 100
    fake_percentage = 100 - real_percentage
    final_label = "Not A Fake News" if real_percentage >= 50 else "Fake News"

    return final_label, real_percentage, fake_percentage

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news = request.form['news']
        label, real_prob, fake_prob = predict_news(news)
        return render_template('result.html', news=news, label=label, real_prob=real_prob, fake_prob=fake_prob)

if __name__ == '__main__':
    app.run(debug=True)
