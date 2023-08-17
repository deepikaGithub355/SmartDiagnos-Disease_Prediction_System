from flask import Flask, render_template, request
app = Flask(__name__)
import pickle
import pandas as pd
import numpy as np


# to open a file,where you stored the pickled data
file = open('model.pkl', 'rb')
model = pickle.load(file)
file.close()

@app.route('/', methods=["GET", "POST"])
def hello_world():
    if request.method == "POST":
        data = request.form
        symps = []
        symps.append(str(data['symp1']))
        symps.append(str(data['symp2']))
        symps.append(str(data['symp3']))
        symps.append(str(data['symp4']))
        symps.append(str(data['symp5']))
        symps.append(str(data['symp6']))
        print(symps)


        df = pd.read_csv("Training.csv.zip")
        df = df.drop("Unnamed: 133", axis=1)
        x = df.drop('prognosis', axis=1)
        

        #tests = input().split(",")
        tests = list(filter(None, symps))
        print(tests)

        data = pd.DataFrame(columns=x.columns.values)

        test_index = [data.columns.get_loc(col) for col in tests]

        # Set specific values to 1 based on user input conditions
        for index in test_index:
            data.loc[0, data.columns[index]] = 1

        # Fill any NaN values with 0 in the 'data' DataFrame
        data.fillna(0, inplace=True)

        test_data = np.array(data)

        # Use the trained Random Forest Classifier to predict the prognosis for the 'test_data'
        res = model.predict(test_data)

        #final result
        finalRes = res[0]
        print(finalRes)

        return render_template('show.html', inf=finalRes)
    return render_template('index.html')
    #return str(finalRes)


if __name__ == "__main__":
    app.run(debug=True)
