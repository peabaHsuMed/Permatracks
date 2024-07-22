import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import csv
from keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import joblib

# Load the data
#data = pd.read_csv('z_39_labeled_data.csv')
#data = pd.read_csv('all_data_z_39_to_79.csv')
#data = pd.read_csv('5_points_labeled_data_from_84_to_144.csv')
data = pd.read_csv('all_data_z_39_to_79_plus_109.csv')


X = data.iloc[:, 1:49].values
y = data.iloc[:, 52:55].values

X_original = X.copy()

print(np.max(X))
print(np.min(X))

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save the scaler
#joblib.dump(scaler, 'scaler.joblib')

#model = load_model('pretrained_model_z_39/model_lr0.001_epochs200_batch16_folds15.h5', custom_objects={'mse': MeanSquaredError()})
#model = load_model('z_39_to_79_model_lr0.001_epochs200_batch16_folds15.h5', custom_objects={'mse': MeanSquaredError()})
#model = load_model('5points_z_84_to_144_model_lr0.001_epochs200_batch16_folds15.h5', custom_objects={'mse': MeanSquaredError()})
model = load_model('z_39_to_79_plus_109_model_lr0.001_epochs200_batch16_folds15.h5', custom_objects={'mse': MeanSquaredError()}) ## bad

#print(model)

error_x_mean = np.zeros((100,1))
error_y_mean = np.zeros((100,1))
error_z_mean = np.zeros((100,1))
euclidean_distance = np.zeros((100,1))
for i in range(100):
    X_train, X_temp, y_train, y_temp, X_original_train, X_original_temp = train_test_split(
        X, y, X_original, test_size=0.3, random_state=i ## make sure every time it splited the same data to get the reproducible results
    )
    X_val, X_test, y_val, y_test, X_original_val, X_original_test = train_test_split(
        X_temp, y_temp, X_original_temp, test_size=0.5, random_state=i
    )

    #model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    predict = model.predict(X_test)
    #print(np.shape(predict)) # (2187,3)
    #print(predict)
    loss = model.evaluate(X_test, y_test)
    euclidean_dist = np.linalg.norm(predict-y_test, axis=1)
    #print(np.shape(euclidean_dist)) # (2187,)
    eucli_df = pd.DataFrame(euclidean_dist)
    abs_error = np.abs(predict-y_test)
    error_df = pd.DataFrame(abs_error)
    #error_df.to_csv('prediction_absolute_error_all_data_39_to_79_2.csv')
    #print(np.abs(predict-y_test))
    error_x_mean[i] = np.mean(error_df.iloc[:,0].values)
    error_y_mean[i] = np.mean(error_df.iloc[:,1].values)
    error_z_mean[i] = np.mean(error_df.iloc[:,2].values)
    euclidean_distance[i] = np.mean(eucli_df.iloc[:,0].values)
    #print("avg x: ", error_x_mean, "avg y: ", error_y_mean, "avg z: ", error_z_mean, "\n")

avg_err = np.concatenate((error_x_mean, error_y_mean, error_z_mean, euclidean_distance), axis=1)
avg_err_df = pd.DataFrame(avg_err)
avg_err_df.to_csv('average error in 3 direction in different random state_39_79_109_plus_euclidean_dist.csv')

# Combine original test features and predictions
result = np.concatenate((X_original_test, predict), axis=1)

# Convert to DataFrame for saving as CSV
columns = [f'Feature_{i+1}' for i in range(X_original_test.shape[1])] + ['Pred_x', 'Pred_y', 'Pred_z']
result_df = pd.DataFrame(result, columns=columns)

#result_df.to_csv('predictions_pretrained_z_39_to_79_model_lr0.001_epochs200_batch16_folds15.csv', index=False)
#print("Predictions with corresponding features saved to csv")
