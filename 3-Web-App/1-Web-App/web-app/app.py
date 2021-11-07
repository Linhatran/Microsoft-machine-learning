import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open('../ufo-model.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()
                    ]  # enumerate string input
    # turn python int list into Numpy array for faster and efficient operations
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = prediction[0]

    countries = ['Australia', 'Canada', 'Germany', 'UK', 'US']
    return render_template('index.html', prediction_text='Possible country: {}'.format(countries[output]))


if __name__ == '__main__':
    app.run(debug=True)
