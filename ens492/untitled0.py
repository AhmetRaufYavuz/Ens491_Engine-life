import keras
import keras.backend as K
from keras.layers.core import Activation
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, LSTM, Concatenate, Input,GRU, Bidirectional
from keras.models import Model
from keras.optimizers import Adam


model1=load_model("model_1.hd5f")

import keras
import keras.backend as K
from keras.layers.core import Activation
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, LSTM, Concatenate, Input,GRU, Bidirectional
from keras.models import Model
from keras.optimizers import Adam


K.clear_session()

nb_features = x_train.shape[2]
nb_out = y_train.shape[1]



input = Input(shape =(None, nb_features))
x = Bidirectional(LSTM(units = 256, return_sequences=True))(input)
x = Dropout(0.2)(x)
x = Bidirectional(LSTM(units=128, return_sequences=True))(x)
x = Dropout(0.2)(x)
x = Bidirectional(LSTM(units=64, return_sequences=True))(x)
x = Dropout(0.2)(x)
x = Bidirectional(LSTM(units=64, return_sequences=False))(x)
x = Dropout(0.2)(x)
output = Dense(units=nb_out)(x)

model1 = Model(inputs = input, outputs = output)
model1.compile(loss='mean_squared_error', optimizer = "rmsprop" ,metrics=['mae'])
print(model1.summary())




from keras.callbacks import ModelCheckpoint, EarlyStopping,ReduceLROnPlateau,CSVLogger

model_name = "model_1.hd5f"
model_path =  model_name
check = ModelCheckpoint(model_path, monitor = "val_loss", save_best_only = True)
lr = ReduceLROnPlateau(monitor = "val_loss", factor = 0.0001, patience = 5, min_lr = 0.0005)

history = model1.fit(x_train, y_train, epochs=80, batch_size=200, validation_split=0.05, 
                     verbose=1,callbacks = [check, lr])

model1=load_model(model_name)

y_pred=model1.predict(x_train,verbose=1)


from sklearn.metrics import mean_absolute_error
mae = str(mean_absolute_error(y_train, y_pred))
print("Train MAE: " + mae)

#Test loss
y_pred = model1.predict(x_test, verbose=1)
mae = str(mean_absolute_error(y_test, y_pred))
print("Test MAE: " + mae)

input = Input(shape =(None, nb_features))
x = LSTM(units = 256, return_sequences=True)(input)
x = Dropout(0.2)(x)
x = LSTM(units=128, return_sequences=True)(x)
x = Dropout(0.2)(x)
x = LSTM(units=64, return_sequences=True)(x)
x = Dropout(0.2)(x)
x = LSTM(units=64, return_sequences=False)(x)
x = Dropout(0.2)(x)
output = Dense(units=nb_out)(x)

model2 = Model(inputs = input, outputs = output)
model2.compile(loss='mean_squared_error', optimizer = "rmsprop" ,metrics=['mae'])
print(model2.summary())
from keras.callbacks import ModelCheckpoint, EarlyStopping,ReduceLROnPlateau,CSVLogger

model_name2 = "model_2.hd5f"
model_path2 =  model_name2
check = ModelCheckpoint(model_path2, monitor = "val_loss", save_best_only = True)
lr = ReduceLROnPlateau(monitor = "val_loss", factor = 0.0001, patience = 5, min_lr = 0.0005)

history = model2.fit(x_train, y_train, epochs=40, batch_size=100, validation_split=0.05, 
                     verbose=1,callbacks = [check, lr])



model2=load_model(model_name2)

y_pred=model2.predict(x_train,verbose=1)


from sklearn.metrics import mean_absolute_error
mae = str(mean_absolute_error(y_train, y_pred))
print("Train MAE: " + mae)

#Test loss
y_pred = model2.predict(x_test, verbose=1)
mae = str(mean_absolute_error(y_test, y_pred))
print("Test MAE: " + mae)