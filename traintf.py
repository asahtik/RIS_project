import tensorflow as tf
import numpy as np
import pandas as pd
import random

SAMPLE_SIZE = 100

filer = pd.read_csv('red.csv')
datar = filer.sample(SAMPLE_SIZE)

fileg = pd.read_csv('green.csv')
datag = fileg.sample(SAMPLE_SIZE)

filey = pd.read_csv('yellow.csv')
datay = filey.sample(SAMPLE_SIZE)

fileb = pd.read_csv('blue.csv')
datab = fileb.sample(SAMPLE_SIZE)

dataset = pd.concat([datar, datag, datay, datab])

train = dataset.sample(frac = 0.7)
y_train = train.pop('label')
test = dataset.drop(train.index)
y_test = test.pop('label')

col_names = ['ind_1', 'ind_2', 'ind_3', 'ind_4', 'ind_5', 'ind_6', 'ind_7', 'ind_8', 'ind_9', 'ind_10', 'ind_11','ind_12', 'ind_13', 'ind_14', 'ind_15', 'ind_16']
feature_columns = []
for name in col_names:
  feature_columns.append(tf.feature_column.numeric_column(name, dtype = tf.float32))

NUM_EXAMPLES = len(y_train)
def make_input_fn(X, y, n_epochs=None, shuffle=True):
  def input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((dict(X), y))
    if(shuffle):
      dataset = dataset.shuffle(NUM_EXAMPLES)
    dataset = dataset.repeat(n_epochs)
    dataset = dataset.batch(NUM_EXAMPLES)
    return dataset
  return input_fn

train_input_fn = make_input_fn(train, y_train)
eval_input_fn = make_input_fn(test, y_test, shuffle = False, n_epochs = 1)

est = tf.estimator.BoostedTreesClassifier(feature_columns, n_batches_per_layer = 1)
est.train(train_input_fn, max_steps=100)

result = est.evaluate(eval_input_fn)
print(pd.Series(result))