#!/usr/bin/python3

import os

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, ReLU, Conv2D, Conv1D, Flatten, Input

from akida import Device, NSoC_v2, AKD1000
import cnn2snn

def main():

  PIXEL_RES = 32

  device = AKD1000()

  coords_ch4 = np.load('coords_ch4.npy')
  color = np.load('result_ch4.npy')

  qcoords = coords_ch4*(PIXEL_RES-1)
  qcoords = qcoords.round()
  qcoords = qcoords.astype(np.uint)

  if os.path.exists('imgs.npy'):
    print('Retreiving imgs from disk...')
    imgs = np.load('imgs.npy')
  else:
    print('Generating images input...')
    empty_img = np.array(PIXEL_RES*[PIXEL_RES*[0]])
    imgs = np.empty((10000, PIXEL_RES, PIXEL_RES))
    for i, p in enumerate(qcoords):
      img = empty_img
      img[p[0], p[1]] = 255
      imgs[i] = img

    imgs = imgs.reshape(10000, PIXEL_RES, PIXEL_RES, 1)
    with open('imgs.npy', 'wb') as f:
      np.save(f, imgs)
    

  model = Sequential()
  model.add(Conv2D(1, (3, 3), input_shape=(PIXEL_RES, PIXEL_RES, 1), name='C2D'))
  model.add(ReLU(name='relu0'))
  model.add(Flatten(name='FLATTEN'))
  model.add(Dense(1000, name='FC1'))
  model.add(ReLU(name='relu1'))
  model.add(Dense(500, name='FC2'))
  model.add(ReLU(name='relu2'))
  model.add(Dense(250, name='FC3'))
  model.add(ReLU(name='relu3'))
  model.add(Dense(100, name='FC4'))
  model.add(ReLU(name='relu4'))
  model.add(Dense(50, name='FC5'))
  model.add(ReLU(name='relu5'))
  model.add(Dense(1, name='FC6'))
  model.add(Activation('sigmoid', name='sigmoid'))

  model = cnn2snn.quantize(model, weight_quantization=4, activ_quantization=4)
  print(model.summary())

  print('Akida Compatibility:', cnn2snn.check_model_compatibility(model, input_is_image=False))

  model.compile(loss='binary_crossentropy', 
                optimizer='adam',
                metrics=['accuracy']
                )
  model.fit(imgs, color, batch_size=64, epochs=10)
  model.evaluate(imgs, color)

  model = cnn2snn.convert(model, input_is_image=False)
  print(model.summary())


  model.map(device)

  qcoords_u8 = qcoords.astype(np.uint8)
  #outputs = model.forward(qcoords_u8)
  #print(outputs)

if __name__ == '__main__':
  main()
