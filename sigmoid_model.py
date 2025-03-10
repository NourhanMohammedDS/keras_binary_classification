import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score, confusion_matrix
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import KFold, cross_val_score
from statsmodels.gam.gam_cross_validation.cross_validators import KFold

# Read the data
data_v = pd.read_csv("visit.csv")
x = data_v[["Time"]]  # Reshape to 2D array
y = data_v[["Buy"]]

# Build the model
model_v = Sequential()
model_v.add(Dense(1, input_shape=(1,), activation="sigmoid"))

# Compile the model
model_v.compile(optimizer=Adam(learning_rate=0.75), loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
model_v.fit(x, y, epochs=50, verbose=1)

# Make predictions
yp = model_v.predict(x)

# Convert predictions to 0 or 1
ypc = (yp >= 0.5).astype(int)

# Evaluate the model
print("Accuracy Score:", accuracy_score(y, ypc))
print("Confusion Matrix:\n", confusion_matrix(y, ypc))

#cross_validation
def get_model():
    model_v = Sequential()
    model_v.add(Dense(1, input_shape=(1,), activation="sigmoid"))
    model_v.compile(optimizer=Adam(learning_rate=0.75), loss="binary_crossentropy", metrics=["accuracy"])
    return model_v
#creat the wrapper
wrapper_model =KerasClassifier(build_fn= get_model() ,epochs=50)
#use KFold for cross-validation
kf = KFold(4)
acc = cross_val_score(wrapper_model, x ,y ,cv=kf)
#cross-validation results
print(acc)
print(acc.mean())
