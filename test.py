import pandas as pd
import tensorflow as tf
import numpy as np

data = pd.read_csv("data.csv")
inp = data.to_numpy()

test = inp[:48]
test_x = test[:,6:24].astype("float32")


model = tf.keras.models.load_model('my_model') 
pred = model.predict(val_x)
print(np.argmax(pred,axis=1)+1)