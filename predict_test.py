import requests

url = "http://localhost:9696/predict"


patient = {
    "gender": "Male",
    "age": "56",                         
    "hypertension": 0,              
    "heart_disease": 1,                    
    "smoking_history": "never",                  
    "bmi": 23.19,                     
    "HbA1c_level": 5.30,                  
    "blood_glucose_level": 103,                        
}

response = requests.post(url, json = patient).json()

print(response)

if response['diabetes']:
    print('Define a treatment for the patient-test.')
else:
    print('The patient seems healthy: no treatment needed.')