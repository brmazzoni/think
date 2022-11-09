#!/usr/bin/python3

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, ReLU, Conv2D, Conv1D

import akida
from akida import Device, NSoC_v2
import cnn2snn

def main():
  device = akida.AKD1000()

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

  model = cnn2snn.quantize(model, weight_quantization=4, activ_quantization=4)
  print(model.summary())

  print('Akida Compatibility:', cnn2snn.check_model_compatibility(model, input_is_image=False))

  model.compile(loss='binary_crossentropy', 
                optimizer='adam',
                metrics=['accuracy']
                )
  model.fit(qcoords, color, batch_size=64, epochs=200)
  model.evaluate(qcoords, color)

  model = cnn2snn.convert(model, input_is_image=False)
  print(model.summary())

  model.map(device)

  qcoords_u8 = qcoords.astype(np.uint8)
  #outputs = model.forward(qcoords_u8)
  #print(outputs)

if __name__ == '__main__':
  main()
