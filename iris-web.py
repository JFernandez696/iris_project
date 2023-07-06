#Import main libraries
import streamlit as st
import pickle
import pandas as pd

#Extrarc the pickle files
with open('lin_reg.pkl', 'rb') as li:
    lin_reg = pickle.load(li)

with open('log_reg.pkl', 'rb') as lo:
    log_reg = pickle.load(lo)

with open('svm_reg.pkl', 'rb') as sv:
    svm_reg = pickle.load(sv)


#Function to classifier plants  
def classify(num):
    if num == 0:
        return 'Setosa'
    elif num == 1:
        return 'Versicolor'
    else:
        return 'Virginica'

def main():
    #Title
    st.title('Modeling of Iris by @JCFE')
    #sidebar title
    st.sidebar.header('User Input Parameters')
    #Function to sidebar parameters
    def user_input_parameters():
        sepal_length = st.sidebar.slider('Sepal Length', 4.3, 7.9, 5.4)
        sepal_width  = st.sidebar.slider('Sepal Width' , 2.0, 4.4, 3.4)
        petal_length = st.sidebar.slider('Petal Length', 1.0, 6.9, 1.5)
        petal_width  = st.sidebar.slider('Petal Width' , 0.1, 2.5, 0.2)

        data = {'sepal_length': sepal_length, 
                'sepal_width': sepal_width,
                'petal_length': petal_length,
                'petal_width' : petal_width,
                }
        features = pd.DataFrame(data, index=[0])

        return features
    
    df = user_input_parameters()
    # Choose the model
    option = ['Linear Regression', 'Logistic Regression', 'SVM']
    model  = st.sidebar.selectbox('Wich model you like to use?', option)

    st.subheader('User Input Parameters')
    st.subheader(model)
    st.write(df)

    if st.button('RUN'):
        if model == 'Linear Regression':
            prediction = lin_reg.predict(df)
            st.success(classify(prediction))

        elif model == 'Logistic Regression':
            prediction = log_reg.predict(df)
            st.success(classify(prediction))

        elif model == 'SVM':
            prediction = svm_reg.predict(df)
            st.success(classify(prediction))



if __name__ == '__main__':
    main()