# import statements
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

# Title of Page
st.write("""
# Simple Iris Flower Prediction App
This app predicts the **Iris flower** type based in the Iris dataset
""")
st.write("Find the dataset [here](https://archive.ics.uci.edu/ml/datasets/iris)")



# Header of sidebar
st.sidebar.header('(Adjust Values Here)')
# st.sidebar.text('Enter Values Here')


# function defining the user input features used to make predictions
def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    #corresponding data
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

# user features that we use to make prediction
df = user_input_features()


# show what the user has inputted
st.subheader('Your chosen values')
st.write(df)
# st.subheader ("These are the values which you have chosen in the sidebar")

# load dataset
iris = datasets.load_iris()
X = iris.data
Y = iris.target

# classifier algorithm
clf = RandomForestClassifier()
clf.fit(X, Y)


# prediction
prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)


# Index of the class labels 
st.subheader('Class labels with corresponding index number')
st.write(iris.target_names)

# class label predicted
st.subheader('Prediction')
st.write(iris.target_names[prediction])
#st.write(prediction)

# probability of each value
st.subheader('Prediction Probability of each type')
st.write(prediction_proba)
