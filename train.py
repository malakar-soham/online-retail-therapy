import tensorflow as tf
import pandas as pd

h=18

def create_model():

    inputs = tf.keras.Input(shape=(h,))
    x = tf.keras.layers.Dense(32, activation='relu')(inputs)
    x1 = tf.keras.layers.Dense(64, activation='relu')(x)
    x2 = tf.keras.layers.Dense(64, activation='relu')(x1)
    outputs = tf.keras.layers.Dense(5, activation='softmax')(x2)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model


data = pd.read_csv("data.csv")
inp = data.to_numpy()
train = inp[48:]
train_x = train[:,6:24]
train_y = train[:,31]
train_y = tf.one_hot(train_y, 6)
train_y = train_y[:,1:].numpy()
train_x = train_x.astype("float32")
val_x, train_x = train_x[-10:], train_x[:-10]
val_y, train_y = train_y[-10:], train_y[:-10]


model = tf.keras.models.load_model('my_model')

#model = create_model()  # uncomment this to create a new model
print(model.summary())

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_x,train_y,epochs=500,batch_size=50, validation_data=(val_x,val_y))
print("Evaluation on Train data")
model.evaluate(train_x,train_y)
print("Evaluation on validation")
model.evaluate(val_x,val_y)

model.save('my_model')