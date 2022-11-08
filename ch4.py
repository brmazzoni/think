#!/usr/bin/python3

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, ReLU
import numpy as np

def main():
  coords_ch4 = np.load('coords_ch4.npy')
  color = np.load('result_ch4.npy')

  qcoords = coords_ch4*15
  qcoords = qcoords.round()


  model = Sequential()

  model.add(Dense(1000, input_shape=(2,), name='FC1'))
  model.add(ReLU(name='relu1'))
  model.add(Dense(500, input_shape=(2,), name='FC2'))
  model.add(ReLU(name='relu2'))
  model.add(Dense(250, input_shape=(2,), name='FC3'))
  model.add(ReLU(name='relu3'))
  model.add(Dense(100, input_shape=(2,), name='FC4'))
  model.add(ReLU(name='relu4'))
  model.add(Dense(50, input_shape=(2,), name='FC5'))
  model.add(ReLU(name='relu5'))
  model.add(Dense(1, input_shape=(2,), name='FC6'))
  model.add(Activation('sigmoid', name='sigmoid'))

  print(model.summary())

  model.compile(loss='binary_crossentropy', 
                optimizer='adam',
                metrics=['accuracy']
                )

  model.fit(qcoords, color, batch_size=64, epochs=200)

  model.evaluate(qcoords, color)


if __name__ == '__main__':
  main()
