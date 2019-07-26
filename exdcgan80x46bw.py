# add: gradient loss to try and control discriminator
import glob
import os
from keras.layers.convolutional import Conv3DTranspose, Conv3D
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, UpSampling3D, Lambda, Cropping3D
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import keras
import time
import cv2
import math
from keras import backend as K
from keras.engine.topology import Layer

import time
llis = ["6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19"]
#llis = ["4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19"]
# Load: contributes the first frame only
in_path = "G:/examples/"
gen_list = llis
disc_list = llis

read = True
training = False
evaluation = True
eval_path = in_path
eval_list = llis
prows = 23
pcols = 40
rows = 45
cols = 80
bfram = 1
afram = 32
channels = 1
alph_const = 100
alphOrg = 75
evalOrg = 60
genBw = 'save/genbwPng2'
ganBw = 'save/ganbwPng2'

prevGenP = 'save/genbwPng'
prevGanP = 'save/ganbwPng'
prev_eval = llis

acc_path = "F:/Images/"

class DCGAN():
  # read: one image
  def readVideo(path, list, read=True, resize=False):
    for l in list:
      if " " not in l:
        with open("Output.txt", "a") as text_file:
          text_file.write(l)
        if read:
          img = os.path.join(path, l, "1.jpg")
          image = (cv2.imread(img))
          image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
          h, w = image.shape
          image = np.reshape(image, (h, w, 1))
          yield (image)/127.5-1
        else:
          if resize == True:
            vid = np.zeros((prows,pcols,afram,channels))
          else:
            vid = np.zeros((rows,cols,afram,channels))
          for a in range(1,33):
            img = os.path.join(path,l, str(a)+".jpg")
            image = (cv2.imread(img))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if resize== True:
              image = cv2.resize(image, (pcols, prows))
            h, w = image.shape
            image = np.reshape(image, (h, w, 1))
            vid[:,:,a-1,:] = (image)/127.5-1
          yield vid
  myConst = readVideo(in_path, gen_list)

  def cevaluate(self):
    for l in eval_list:
      print(l)
      if " " not in l:
        img =os.path.join(in_path, l, "1.jpg")
        print(img)  
        image = (cv2.imread(img))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = image.shape
        global alphOrg
        alph = evalOrg/alph_const
        alph = np.asarray(np.repeat(alph, 5))
        
        image = (np.reshape(image, (h, w, 1)))/127.5-1
        imgs=[]
        #shrinked images
        previmgs = np.zeros((5, prows, pcols, bfram, channels))
        prev_gen = next(self.myConst)
        for i in range(1):
          # 45, 80, 1 : input is this size, and network builds on it
          shrinked = cv2.resize(prev_gen, (pcols, prows))
          # 23, 40
          resized_img = cv2.resize(prev_gen, (cols, rows))
          #print(resized_img.shape)
          imgs.append(np.reshape(resized_img, (rows, cols, 1, channels)))
          previmgs[i, :,:,0,0] = shrinked#.append(np.reshape(shrinked, (1, prows, pcols, bfram, channels)))
        # 23, 40, 1, 1
        prevGenRes = np.asarray((self.prevGen).predict_on_batch(np.asarray(previmgs)))
        # 5 23 40 32 1
        timesmp1 = int(time.time())
        os.makedirs(eval_path + str(timesmp1))
        for frame in range(1,33):
          predFrame = ((prevGenRes[0,:,:,frame-1,:]+1)*127.5).astype(int)
          smth = cv2.imwrite(eval_path + str(timesmp1) +"/"+str(frame)+'.png',predFrame)
        
        predicted = self.generator.predict([imgs, prevGenRes, alph])
        # 5, 46, 80, 32, 1
        timesmp = int(time.time())+1
        os.makedirs(eval_path + str(timesmp), exist_ok=True)

        for frame in range(1,33):
          predFrame = ((predicted[0,:,:,frame-1,:]+1)*127.5).astype(int)
          # 46, 80
          smth = cv2.imwrite(eval_path + str(timesmp) +"/"+str(frame)+'.png',predFrame)
      yield
    
  def __init__(self, read=False):
    self.pre_shape = (rows, cols,bfram, channels)
    self.prev_shape = (prows, pcols, bfram, channels)
    self.generated_shape = (rows, cols, afram, channels)
    self.prev_gen = (prows, pcols, afram, channels)
    self.bs = 5
    #sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    optimizer = Adam(0.0005, 0.5, 0.9)
    # previous network
    self.prevGen =keras.models.load_model(prevGenP, compile=False)
    self.prevGan =keras.models.load_model(prevGanP, compile=False)
    
    #Load model
    if read == True:
      (self.discriminator) =keras.models.load_model(ganBw, compile=False)
      (self.generator) =keras.models.load_model(genBw, compile=False)
    if read == False:
      # Build and compile the discriminator
      self.discriminator = self.build_discriminator()
    # changes: manually compile discriminator if case is 'read'
    self.discriminator.compile(loss='mse',
          optimizer=optimizer,
          metrics=['accuracy'])
    
    if read == False:
      # Build the generator
      self.generator = self.build_generator()

    z = Input(shape=self.pre_shape)
    in_mask = Input(shape=self.prev_gen)
    #alpha = Input(dtype='float32')
    # input 6, 7, 8, 9?
    mask, alpha, mask2, alpha2 = Input(shape=(1,)), Input(shape=(1,)), Input(shape=(1,)), Input(shape=(1,))
    img = self.generator([z, in_mask, alpha])

    self.generator.compile(loss='mse',
      optimizer=optimizer,
      metrics=['accuracy'])
    
    self.discriminator.trainable = False
    valid = self.discriminator([img, alpha, alpha])
    
    self.combined = Model([z, in_mask, alpha], valid)
    self.combined.compile(loss='mse', optimizer=optimizer)
    
  
  def build_generator(self):
    i = Input(shape=self.pre_shape)
    mask = Input(shape=self.prev_gen)
    alph = Input(shape=(1,))
    inmask = (UpSampling3D(size=(2, 2, 1), data_format="channels_last", input_shape=self.pre_shape))(mask)
    m = (UpSampling3D(size=(1, 1, 2), data_format="channels_last", input_shape=self.pre_shape))(i)
    m = (Conv3D(256, kernel_size=(3,3,3), padding="same", input_shape=self.pre_shape, data_format="channels_last"))(m)
    m = (BatchNormalization(momentum=0.5))(m)
    m = (Activation("relu"))(m)
    
    m = (Conv3D(128, kernel_size=(3,3,3), padding="same", input_shape=self.pre_shape, data_format="channels_last"))(m)
    m = (BatchNormalization(momentum=0.5))(m)
    m = (Activation("relu"))(m)
    m = (UpSampling3D(size=(1, 1, 2), data_format="channels_last"))(m)
    m = (Conv3D(64, kernel_size=(3,3,3), padding="same", input_shape=self.pre_shape ,data_format="channels_last"))(m)
    m = (BatchNormalization(momentum=0.5))(m)
    m = (Activation("relu"))(m)
    m = (UpSampling3D(size=(1, 1, 2), data_format="channels_last"))(m)
    m = (Conv3D(32, kernel_size=(3,3,3), padding="same", input_shape=self.pre_shape, data_format="channels_last"))(m)
    m = (BatchNormalization(momentum=0.5))(m)
    m = (Activation("relu"))(m)
    m = (UpSampling3D(size=(1, 1, 2), data_format="channels_last"))(m)
    m = (Conv3D(16, kernel_size=(3,3,3), padding="same", input_shape=self.pre_shape, data_format="channels_last"))(m)
    m = (BatchNormalization(momentum=0.5))(m)
    m = (Activation("relu"))(m)
    m = (UpSampling3D(size=(1, 1, 2), data_format="channels_last"))(m)
    m = (Conv3D(8, kernel_size=(3,3,3), padding="same", input_shape=self.pre_shape, data_format="channels_last"))(m)
    m = (BatchNormalization(momentum=0.5))(m)
    m = (Activation("relu"))(m)
    m = (Conv3D(1, kernel_size=(3,3,3), padding="same", input_shape=self.pre_shape, data_format="channels_last"))(m)
    m = (Activation("tanh"))(m)
    
    inmask = Cropping3D(cropping=((0, 1), (0, 0), (0, 0)), data_format="channels_last")(inmask)
    added = Lambda(lambda x: x[2][0]*(x[0])+(1-x[2][0])*(x[1]))([m, inmask, alph])
    model = Model(inputs=[i, mask, alph], outputs=added)
    model.summary()
    return model
  def build_discriminator(self):
    gen = Input(shape=self.generated_shape, name="di_80")
    mask = Input(shape=(1,), name="di_40")
    alph = Input(shape=(1,), name="di_alpha")
    m = (Conv3D(64, 3, input_shape=self.generated_shape,strides=2, padding='same', data_format="channels_last"))(gen)
    m = (LeakyReLU(alpha=0.2))(m)
    m = (BatchNormalization(momentum=0.8))(m)
    m = (Conv3D(32, 3, strides=2, padding='same', data_format="channels_last"))(m)
    m = (LeakyReLU(alpha=0.2))(m)
    m = (BatchNormalization(momentum=0.8))(m)
    m = (Conv3D(8, 3, strides=2, padding='same', data_format="channels_last"))(m)
    m = (LeakyReLU(alpha=0.2))(m)
    m = (BatchNormalization(momentum=0.8))(m)
    m = (Conv3D(1, 3, strides=2, padding='same', data_format="channels_last"))(m)
    m = (LeakyReLU(alpha=0.2))(m)
    m = (BatchNormalization(momentum=0.8))(m)
    m = (Flatten())(m)
    m = (Dense(1))(m)
    added = Lambda(lambda x: x[2]*(x[0])+(1-x[2])*(x[1]))([m, mask, alph])
    model = Model(inputs=[gen, mask, alph], outputs=added)
    #model.summary()
    return model

  def train(self, epochs, save_interval=50):
    func = self.cevaluate()
    tconstant = 1
    ctr = 0
    gen_batch = readVideo(in_path, gen_list)
    dis_batch = readVideo(in_path, disc_list, read=False)


    prev_gen = readVideo(in_path, gen_list)
    prev_dis = readVideo(in_path, disc_list, read=False, resize=True)
    
    fake_predictions = []
    real_predictions = []
    for epoch in range(epochs):
      global alphOrg
      #if epoch > 200:
      #  alph = math.log((epoch*5), 0.019869)+2.51281
      #else: 
      alph = alphOrg/alph_const
      imgs = []
      videos = []
      previmgs = []
      prevvid = []
      # batch of prev. images
      for i in range(self.bs):
        nextimg = next(prev_gen)
        nextdis = next(prev_dis)
        shrinked = cv2.resize(nextimg, (pcols, prows))

        
        previmgs.append(np.reshape(shrinked, self.prev_shape))
        prevvid.append(nextdis)
        
      # evaluate fake & real from prev network:
      prevGenRes = (self.prevGen).predict_on_batch(np.asarray(previmgs))
      prevGanRes = (self.prevGan).predict_on_batch(np.asarray(prevGenRes))
      prevGanResReal = (self.prevGan).predict_on_batch(np.asarray(prevvid))
      
      for i in range(self.bs):
        imgs.append(np.reshape(next(gen_batch), (rows, cols, 1, channels)))
        videos.append(next(dis_batch))
      valid = np.ones((self.bs, 1))
      fake = np.zeros((self.bs, 1))
      
      
      # transformation
      videos = np.asarray(videos)
      # need to duplicate 5 for a batch.....: (
      alph = np.asarray(np.repeat(alph, self.bs))
      imgs = np.asarray(imgs)
      prevGenRes = np.asarray(prevGenRes)
      genVid = self.generator.predict([imgs, prevGenRes, alph])
      prediction1 = self.discriminator.predict_on_batch([np.asarray(genVid), prevGanRes, alph])
      print("Predict gen", prediction1)
      fake_predictions = fake_predictions + prediction1
      prediction2 = self.discriminator.predict_on_batch([np.asarray(videos), prevGanResReal, alph])
      print("Predict real", prediction2)

      real_predictions = real_predictions + prediction2
      
      d_loss_real = self.discriminator.train_on_batch([genVid, prediction1, alph], fake)
      d_loss_fake = self.discriminator.train_on_batch([videos, prediction2, alph], valid)
      

      #  Train Generator
      g_loss = self.combined.train_on_batch([np.asarray(imgs), prevGenRes, alph], valid)
      print("Real loss", d_loss_real, "Fake loss", d_loss_fake, "G_Loss",str(g_loss))
      print ("%d " % (epoch))
      ctr = ctr + 1
      if ctr %save_interval == 0:
        g_loss = self.combined.train_on_batch([np.asarray(imgs), prevGenRes, alph], valid)
        print ("%d [G loss: %f]" % (epoch, g_loss))
        (self.generator).save(genBw)
        (self.discriminator).save(ganBw)
        next(func)
        tconstant = tconstant + 1
if __name__ == '__main__':
  tf.reset_default_graph()
  if not os.path.isfile('./'+ganBw) or read == False:
    read = False

  dcgan = DCGAN(read=read)
  if training == True:
    dcgan.train(epochs=1000, save_interval=5)
  
  if evaluation:
    func = dcgan.cevaluate()
    for i in range(17):
      print("done")
      next(func)
      time.sleep(1)