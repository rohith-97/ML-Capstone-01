import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder



n_estimators, max_depth, min_samples_leaf = 40, 10, 1
n_splits = 6
output_file = f'rf_model_{n_estimators}_trees_depth_{max_depth}_min_samples_leaf_{min_samples_leaf}.bin'

df = pd.read_csv("diabetes_prediction_dataset.csv")

categorical_features = []
numerical_features = []

for col in df.columns:
    if col == "diabetes":
        continue  # do not treat target as a feature

    if df[col].nunique() > 6:
        numerical_features.append(col)
    else:
        categorical_features.append(col)

print("Categorical Features:", categorical_features)
print("Numerical Features:", numerical_features)

encoder = LabelEncoder()
df_encoded = df.copy()
cols_to_encode = ['gender', 'hypertension', 'heart_disease', 'smoking_history']
for col in cols_to_encode:
    df_encoded[col] = encoder.fit_transform(df_encoded[col])

# Splitting into full train and test
df_full_train, df_test = train_test_split(df, test_size = 0.2, random_state = 42)

# Splitting into train and test
df_train, df_val = train_test_split(df_full_train, test_size = 0.25, random_state = 42)

df_train = df_train.reset_index(drop = True)
df_test = df_test.reset_index(drop = True)
df_val = df_val.reset_index(drop = True)

y_train = df_train.diabetes.values
y_test = df_test.diabetes.values
y_val = df_val.diabetes.values

del df_train["diabetes"]
del df_test["diabetes"]
del df_val["diabetes"]

def train(df_train, y_train, n_estimators = 40, max_depth = 10, min_samples_leaf = 25):
    train_dicts = df_train[categorical_features + numerical_features].to_dict(orient = 'records')
    
   
    One_Hot_encoder = DictVectorizer(sparse = False)    # Initialize One-Hot-Encoder (vectorizer)
                                                        # One-Hot-Encoder training and train data encoding
    X_train = One_Hot_encoder.fit_transform(train_dicts)

                                                        # Initialize random forest model
    rf = RandomForestClassifier(n_estimators = n_estimators,
                                max_depth = max_depth,
                                min_samples_leaf = min_samples_leaf,
                                random_state = 42,
                                n_jobs = -1)
    rf.fit(X_train, y_train)
    return One_Hot_encoder, rf
                                        # Function to make predictions with a random forest classifier
def predict(df, One_Hot_encoder, rf):
    dicts = df[categorical_features + numerical_features].to_dict(orient = 'records')

   
    X = One_Hot_encoder.transform(dicts)       # One-Hot-Encoding
                                            
    y_pred = rf.predict(X)                     # Make predictions 
    
    return y_pred

print(f"Performing KFold Cross-Validation")
kfold = KFold(n_splits = n_splits, shuffle = True, random_state = 1)
scores = []
fold = 0
for train_idx, val_idx in kfold.split(df_full_train):             # Select train and validation data
    
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]


    y_train = df_train.diabetes.values                              # Select target variables
    y_val = df_val.diabetes.values

                                                                  # Train model
    One_Hot_encoder, rf = train(df_train, y_train)
                                                                  # Make predictions
    y_pred = predict(df_val, One_Hot_encoder, rf)

    
    Accuracy_score = round(100 * (y_pred == y_val).mean(), 2)
    scores.append(Accuracy_score)
    print(f"Accuracy on fold {fold} is {Accuracy_score} %.")
                                                                   # Increment number of fold
    fold += 1
    
print("Validation results:")
print('Accuracy_score mean = %.2f, Accuracy_score std = +- %.2f' % (np.mean(scores), np.std(scores)))

One_Hot_encoder, rf = train(df_full_train[categorical_features + numerical_features], df_full_train.diabetes,
                            n_estimators = n_estimators, max_depth = max_depth,
                            min_samples_leaf = min_samples_leaf)
y_pred = predict(df_test, One_Hot_encoder, rf)
print('Optimal model accuracy = %.2f.' % (100 * (y_pred == y_test).mean()))

with open(output_file, 'wb') as f_out:
    print("Storing the model into a file")
    pickle.dump((One_Hot_encoder, rf), f_out)
    
print(f"The model is saved to {output_file}.")