import pandas as pd
from model import nlp

#Read the target excel file
file = pd.read_csv(r"C:/Users/uchei/Downloads/Summer Internship - Homework Exercise.csv")
#Slice the data for validation only
valid_data = file[100:200]
#Select the targeted column
valid_data = valid_data['transaction_descriptor']

#Performing validation
#Convert the Pandas Series to string
validation_text = valid_data.to_string()
for i in validation_text1:
    doc1 = nlp(validation_text1)
#Get results using the model
for ent in doc1.ents:
    print(ent)
    
#Import testing data from the file
test_data = file[200:300]
#Specify the column to run the trained model on
test_data = test_data['transaction_descriptor']

#Convert the pandas Series to string so the model can recognize it
testing_text = test_data.to_string()
for i in testing_text:
    doc2 = nlp(testing_text)
#Print extracted entities
for ent in doc2.ents:
    print(ent)
