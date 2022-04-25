from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler #normalize and scale feature data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as ny
import pandas as pd
import seaborn as sns
df=pd.read_csv('board_data.csv')
df.head()
x=df[['(0;0)', '(1;0)', '(2;0)', '(3;0)', '(0;1)', '(1;1)', '(2;1)', '(3;1)', '(0;2)', '(1;2)', '(2;2)', '(3;2)', '(0;3)', '(1;3)', '(2;3)', '(3;3)']].values
y=df[['s(0;0)', 's(1;0)', 's(2;0)', 's(3;0)', 's(0;1)', 's(1;1)', 's(2;1)', 's(3;1)', 's(0;2)', 's(1;2)', 's(2;2)', 's(3;2)', 's(0;3)', 's(1;3)', 's(2;3)', 's(3;3)']].values
n = x.shape[1] # n is amount of features being processed
x_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=102)
scaler = MinMaxScaler()
scaler.fit(x_train)
model = Sequential()
model.add(Dense((10 * n), activation='relu'))  # 1st layer -- input layer
model.add(Dense((20 * n), activation='relu'))  # 2nd layer
model.add(Dense((10 * n), activation='relu'))  # 3rd layer
model.add(Dense((5 * n), activation='sigmoid'))  # 4th layer
model.add(Dense(n))  # output (may change n to 1 if we decide to only have 1 output. Currently set up to output a percent chance of a bomb at a given space)
model.compile(optimizer='adam', loss='binary_crossentropy')  # main function to imporve nn
model.fit(x=x_train, y=y_train,epochs=300)  # epochs -> number of pass over the entired dataset ->this is where neural network is run
loss_df = pd.DataFrame(model.history.history)
model.evaluate(X_test, y_test, verbose=0) # returns mean square error
model.evaluate(x_train, y_train, verbose=0)
test_predictions = model.predict(X_test)
test_predictions = pd.Series(test_predictions.flatten()) #converting to dataframe
pred_df = pd.DataFrame(y_test, columns=['orig s(0;0)', 'orig s(1;0)', 'orig s(2;0)', 'orig s(3;0)', 'orig s(0;1)', 'orig s(1;1)', 'orig s(2;1)', 'orig s(3;1)', 'orig s(0;2)', 'orig s(1;2)', 'orig s(2;2)', 'orig s(3;2)', 'orig s(0;3)', 'orig s(1;3)', 'orig s(2;3)', 'orig s(3;3)'])
pred_df = pd.concat([pred_df, test_predictions], axis=1)
pred_df.columns = ['orig s(0;0)', 'orig s(1;0)', 'orig s(2;0)', 'orig s(3;0)', 'orig s(0;1)', 'orig s(1;1)', 'orig s(2;1)', 'orig s(3;1)', 'orig s(0;2)', 'orig s(1;2)', 'orig s(2;2)', 'orig s(3;2)', 'orig s(0;3)', 'orig s(1;3)', 'orig s(2;3)', 'orig s(3;3)', 'Predictions']
loss_df.plot()
#Saves model
from tensorflow.keras.models import load_model
model.save('my_minesweeper_model.h5')
#loads model
my_model = load_model('my_minesweeper_model.h5')
