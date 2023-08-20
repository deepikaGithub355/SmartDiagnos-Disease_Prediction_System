import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

 
if __name__ == "__main__":
    df = pd.read_csv("Training.csv.zip")

    df = df.drop("Unnamed: 133", axis=1)

    x = df.drop('prognosis', axis=1)  # input variable
    y = df['prognosis']  # output variable
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)  # variables to store the split datasets

    # Random Forest Classifier model
    model = RandomForestClassifier()

    # model fitting
    model = model.fit(x_train, y_train)


    # open a file, where you can store the data
    file = open('model.pkl', 'wb')

    # dump information to that file
    pickle.dump(model, file)

    file.close()
        