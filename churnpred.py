import streamlit as st
import pickle
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder, StandardScaler
le =LabelEncoder()
ss =StandardScaler()

#load the model
model =pickle.load(open("7_logistic_model.pkl","rb"))

#creating the web app
st.title("Logistic Regression for churn prediction")
gender= st.selectbox('Select Gender',options=['Male','Female'] ) #selection box with option
SeniorCitizen =st.selectbox('Are u a senior citizen ?',options=['Yes','No'])
Partner =st.selectbox('Do you have partner ?',options=['Yes','No'])
Dependents =st.selectbox('Are you dependent on other?',options=['Yes','No'])
tenure =st.text_input('Enter your tenure?') #to enter the value creating blank space
PhoneService =st.selectbox('Do you have phone service?',options=['Yes','No'])
MultipleLines =st.selectbox('Do you have multilines services?',options=['Yes','No','no phone service'])
Contract =st.selectbox('Your contracts?',options=['One year','Two year','Month_to_month'])
TotalCharges =st.text_input("Enter your total charges?")
#-----------------------------------------------------------------------------------------------------------------------------------
#helper functions (copied from notebook where we did the classification )

#now whenever we pass the data or entries it first go through this funct
#and all these operation will be applied on that particular entered value

def predictive(gender,Seniorcitizen,Partner,Dependents,tenure,Phoneservice,multiline,contract,totalcharge):
    #passing the param value to the required columns of dataset in form column:value
    data ={
        'gender':[gender],
        'SeniorCitizen':[Seniorcitizen],
        'Partner':[Partner],
        'Dependents':[Dependents],
        'tenure':[tenure],
        'PhoneService':[Phoneservice],
        'MultipleLines':[multiline],
        'Contract':[contract],
        'TotalCharges':[totalcharge]
    }
    df1 =pd.DataFrame(data) #converting the data that we pass into dataframe

    #now we will apply all those operation that we did earlier on this dataframe
    categorical_columns =['gender','SeniorCitizen',	'Partner','Dependents',	'tenure','PhoneService','MultipleLines','Contract','TotalCharges']
    for col in categorical_columns:
        df1[col] = le.fit_transform(df1[col]) #label encoding the columns

    df1 =ss.fit_transform(df1) #standardized the values
    result =model.predict(df1).reshape(1,-1) #here we predict the result using the model that we loaded here
    return result[0]
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
#adding tips with the result 
#Tips for Churn Prevention (adding few tips in results)
churn_tips_data = {
    "Tips for Churn Prevention": [
        "Identify the Reasons: Understand why customers or employees are leaving. Conduct surveys, interviews, or exit interviews to gather feedback and identify common issues or pain points.",
        "Improve Communication: Maintain open and transparent communication channels. Address concerns promptly and proactively. Make sure customers or employees feel heard and valued.",
        "Enhance Customer/Employee Experience: Focus on improving the overall experience. This could involve improving product/service quality or creating a more positive work environment for employees.",
        "Offer Incentives: Provide incentives or loyalty programs to retain customers. For employees, consider benefits, bonuses, or career development opportunities.",
        "Personalize Interactions: Tailor interactions and offers to individual needs and preferences. Personalization can make customers or employees feel more connected and valued.",
        "Monitor Engagement: Continuously track customer or employee engagement. For customers, this might involve monitoring product usage or website/app activity. For employees, assess job satisfaction and engagement levels.",
        "Predictive Analytics: Use data and predictive analytics to anticipate churn. Machine learning models can help identify patterns and predict which customers or employees are most likely to churn.",
        "Feedback Loop: Create a feedback loop for ongoing improvement. Regularly seek feedback, analyze it, and use it to make informed decisions and changes.",
        "Employee Training and Development: Invest in training and development programs for employees. Opportunities for growth and skill development can improve job satisfaction and loyalty.",
        "Competitive Analysis: Stay aware of what competitors are offering. Ensure your products, services, and workplace environment remain competitive in the market."
    ]
}

#Tips for customer retention (Not churning) --> these instruction are from gpt
retention_tips_data = {
    "Tips for Customer Retention": [
        "Provide Exceptional Customer Service: Ensure that customers receive excellent customer service and support.",
        "Create Loyalty Programs: Reward loyal customers with discounts, special offers, or exclusive access to products/services.",
        "Regularly Communicate with Customers: Keep customers informed about updates, new features, and promotions.",
        "Offer High-Quality Products/Services: Consistently deliver high-quality products or services that meet customer needs.",
        "Resolve Issues Quickly: Address customer concerns and issues promptly to maintain their satisfaction.",
        "Build Strong Customer Relationships: Develop strong relationships with customers by understanding their needs and preferences.",
        "Provide Value: Offer value-added services or content that keeps customers engaged and interested.",
        "Simplify Processes: Make it easy for customers to do business with you. Simplify processes and reduce friction.",
        "Stay Responsive: Be responsive to customer inquiries and feedback, even on social media and review platforms.",
        "Show Appreciation: Express gratitude to loyal customers and acknowledge their continued support."
    ]
}

#creating Dataframes for the tips
churn_tips_df =pd.DataFrame(churn_tips_data)
retention_tips_data =pd.DataFrame(retention_tips_data)
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#button

#then it will predict the churn after the operation is performed on entered values 
if st.button("Predict"):
    result =predictive(gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,Contract,TotalCharges)
    if result ==1:
        st.title("Churn")
        st.write("Here are 10 tips for churn Prevention:") #after predicting the result it will also show the tips accordingly
        st.dataframe(churn_tips_data,height=400,width=600)
    else :
        st.title("Not Churn")
        st.write("Here are 10 tips for Customer Retention (Not Churning)")
        st.dataframe(retention_tips_data,height=400,width=600)




#---------------------------------------------------------------
#This is the web app for the churn prediction using logistic regression