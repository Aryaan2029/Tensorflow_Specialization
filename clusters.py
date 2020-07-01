import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np

train_df = pd.read_csv("./data/train.csv")
colors_list = train_df.color.unique()
colors_dict = {}
count = 0
for i in colors_list:
    colors_dict[i] = count
    count += 1

train_df['color'] = train_df.color.apply(lambda x : colors_dict[x])

np.random.shuffle(train_df.values)

model = keras.Sequential([
    keras.layers.Dense(32, input_shape = (2,), activation = 'relu'),
    keras.layers.Dense(32, activation = 'relu'),
    keras.layers.Dense(6, activation = 'sigmoid')])

model.compile(optimizer = 'adam', loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

x = np.column_stack((train_df.x.values, train_df.y.values))
model.fit(x, train_df.color.values, batch_size= 4, epochs = 10)


test_df = pd.read_csv("./data/test.csv")
test_df['color'] = test_df.color.apply(lambda x : colors_dict[x])
np.random.shuffle(test_df.values)
test_x = np.column_stack((test_df.x.values, test_df.y.values))

model.evaluate(test_x, test_df.color.values)

print(model.predict(np.array([[0.7, 3]])))
