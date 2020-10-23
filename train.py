# Importing Libraries

import logging
import os
import sys
import PIL
import numpy as np
import tensorflow 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_probability as tfp

from misc import *
from PIL import Image
from numpy import asarray
from tensorflow.keras import datasets, layers, models

class Wasserstein_GAN(object):

  def __init__(self, **kw):
    self.images = []
    self._height               = retrieve_kw(kw, 'height',               128                                                       )
    self._width                = retrieve_kw(kw, 'width',                128                                                       )
    self._max_epochs           = retrieve_kw(kw, 'max_epochs',           1000                                                       )
    self._batch_size           = retrieve_kw(kw, 'batch_size',           32                                                       )
    self._n_features           = retrieve_kw(kw, 'n_features',           NotSet                                                     )
    self._n_critic             = retrieve_kw(kw, 'n_critic',               0                                                        )
    self._result_file          = retrieve_kw(kw, 'result_file',          "check_file"                                                     )
    self._save_interval        = retrieve_kw(kw, 'save_interval',        200                                                      )
    self._use_gradient_penalty = retrieve_kw(kw, 'use_gradient_penalty', True                                                       )
    self._verbose              = retrieve_kw(kw, 'verbose',              True                                                       )
    self._gen_opt              = retrieve_kw(kw, 'gen_opt',              tf.optimizers.Adam(lr=1e-4, beta_1=0.5, decay=1e-4 )       )
    self._critic_opt           = retrieve_kw(kw, 'critic_opt',           tf.optimizers.Adam(lr=1e-4, beta_1=0.5, decay=1e-4 )       )
    self._tf_call_kw           = retrieve_kw(kw, 'tf_call_kw',           {}                                                         )
    self._grad_weight          = tf.constant( retrieve_kw(kw, 'grad_weight',          10.0                                          ) )
    self._latent_dim           = tf.constant( retrieve_kw(kw, 'latent_dim',           100                                          ) )
    self._leaky_relu_alpha     = retrieve_kw(kw, 'leaky_relu_alpha',     0.3                                                    )

    # Initialize discriminator and generator networks
    self.critic = self._build_critic()
    self.generator = self._build_generator()

  def latent_dim(self):
    return self._latent_dim

  @tf.function
  def latent_log_prob(self, latent):
    prior = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(self._latent_dim),
                                                     scale_diag=tf.ones(self._latent_dim))
    return prior.log_prob(latent)

  @tf.function
  def wasserstein_loss(self, y_true, y_pred):
    return tf.reduce_mean(y_true) - tf.reduce_mean(y_pred)

  @tf.function
  def sample_latent_data(self, nsamples):
    return tf.random.normal((nsamples, self._latent_dim))

  @tf.function
  def transform(self, latent):
    return self.generator( latent, **self._tf_call_kw)

  @tf.function
  def generate(self, nsamples):
    return self.transform( self.sample_latent_data( nsamples ))

  def train(self, train_data):
    if self._n_features is NotSet:
      self._n_features = train_data.shape[1]
    if self._verbose: print('Number of features is %d.' % self._n_features )
    gpus = tf.config.experimental.list_physical_devices('GPU')
    n_gpus = len(gpus)
    if self._verbose: print('This machine has %i GPUs.' % n_gpus)

    train_dataset = tf.data.Dataset.from_tensor_slices( train_data ).batch( self._batch_size, drop_remainder = True )

    # checkpoint for the model
    checkpoint_maker = tf.train.Checkpoint(generator_optimizer=self._gen_opt,
        discriminator_optimizer=self._critic_opt,
        generator=self.generator,
        discriminator=self.critic
    )if self._result_file else None

    # containers for losses
    losses = {'critic': [], 'generator': [], 'regularizer': []}
    critic_acc = []

    #reduce_lr = ReduceLROnPlateau(monitor='loss', patience=100, mode='auto',
    #                              factor=factor, cooldown=0, min_lr=1e-4, verbose=2)
    #model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=callback_list,
    #          class_weight=class_weight, verbose=2, validation_data=(X_test, y_test))

    updates = 0; batches = 0;
    for epoch in range(self._max_epochs):
      for sample_batch in train_dataset:
        if self._n_critic and (updates % self._n_critic):
          # Update only critic
          critic_loss, reg_loss, gen_loss = self._train_critic(sample_batch) + (np.nan,)
        if not(self._n_critic) or not(updates % self._n_critic):
          # Update critic and generator
          critic_loss, gen_loss, reg_loss = self._train_step(sample_batch)
        losses['critic'].append(critic_loss)
        losses['generator'].append(gen_loss)
        losses['regularizer'].append(reg_loss)
        updates += 1
        #perc = np.around(100*epoch/self._max_epochs, decimals=1)

        # Save current model
        #if checkpoint_maker and not(updates % self._save_interval):
        #  checkpoint_maker.save(file_prefix=self._result_file)
        #  pass
        # Print logging information
        if self._verbose and not(updates % self._save_interval):
          perc = np.around(100*epoch/self._max_epochs, decimals=1)
          print('Epoch: %i. Updates %i. Training %1.1f%% complete. Critic_loss: %.3f. Gen_loss: %.3f. Regularizer: %.3f'
               % (epoch, updates, perc, critic_loss, gen_loss, reg_loss ))
          img_geradora = wgan.generate(1)
          self.images.append(img_geradora)

    checkpoint_maker.save(file_prefix=self._result_file)
    self.save( overwrite = True )
    return losses

  def save(self, overwrite = False ):
    self.generator.save_weights( self._result_file + '_generator', overwrite )
    self.critic.save_weights( self._result_file + '_critic', overwrite )

  def load(self, path ):
    self.generator.load_weights( path + '_generator' )
    self.critic.load_weights( path + '_critic' )

  def _build_critic(self):
    ip = layers.Input(shape=(self._height,self._width,1))
    # TODO Add other normalization scheme as mentioned in the article
    # Input (None, 3^2*2^5 = 1 day = 288 samples, 1)
    y = layers.Conv2D(256, (5,5), strides=(2,2), padding='same', kernel_initializer='he_uniform', data_format='channels_last', input_shape=(self._height,self._width,1))(ip)
    #y = layers.BatchNormalization()(y)
    y = layers.Activation('relu')(y)
    y = layers.Dropout(rate=0.3, seed=1)(y)
    # Output (None, 3^2*2^3, 64)
    y = layers.Conv2D(128, (5,5), strides=(2,2), padding='same', kernel_initializer='he_uniform')(y)
    #y = layers.BatchNormalization()(y)
    y = layers.Activation('relu')(y)
    y = layers.Dropout(rate=0.3, seed=1)(y)
    # Output (None, 3^2*2^3, 64)
    y = layers.Conv2D(64, (5,5), strides=(2,2), padding='same', kernel_initializer='he_uniform')(y)
    #y = layers.BatchNormalization()(y)
    y = layers.Activation('relu')(y)
    y = layers.Dropout(rate=0.3, seed=1)(y)
    # Output (None, 3^2*2, 128)
    y = layers.Flatten()(y)
    # Output (None, 3*256)
    #out = layers.Dense(nb_class, activation='sigmoid')(y)
    out = layers.Dense(1, activation='linear')(y)
    # Output (None, 1)
    model = tf.keras.Model(ip, out)
    if self._verbose: model.summary()
    model.compile()
    #y = layers.GlobalAveragePooling1D()(y)
    return model



  def _build_generator(self):
    ip = layers.Input(shape=(self._latent_dim,))
    # Input (None, latent space (100?) )
    y = layers.Dense(units=16*16*32, input_shape=(self._latent_dim,))(ip)
    # Output (None, 64*3^2 )
    y = layers.Reshape(target_shape=(16,16, 32))(y)
    #y = layers.BatchNormalization()(y)
    #y = layers.LeakyReLU(alpha=self._leaky_relu_alpha)(y)
    #y = layers.UpSampling1D()(y)
    # Output (None, 3^2*2, 64)
    y = layers.Conv2DTranspose(64, (4,4), strides=(2,2), padding='same', kernel_initializer='he_uniform')(y)
    y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU(alpha=self._leaky_relu_alpha)(y)
    y = layers.Dropout(rate=0.3)(y)
    #y = layers.UpSampling1D(size=2*2)(y)
    # Output (None, 3^2*2^3, 128)
    y = layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer='he_uniform')(y)
    y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU(alpha=self._leaky_relu_alpha)(y)
    y = layers.Dropout(rate=0.3)(y)
    #y = layers.UpSampling1D(size=2*2)(y)
    # Output (None, 3^2*2^5, 256)
    y = layers.Conv2DTranspose(256, (4,4), strides=(2,2), padding='same', kernel_initializer='he_uniform')(y)
    y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU(alpha=self._leaky_relu_alpha)(y)
    y = layers.Dropout(rate=0.3)(y)
    # Output (None, 3^2*2^5, 64)
    out = layers.Conv2DTranspose(1, (4,4), strides=(1,1), padding='same', kernel_initializer='he_uniform', activation = 'tanh')(y)
    # Output (None, 3^2*2^5, 1)
    model = tf.keras.Model(ip, out)
    if self._verbose: model.summary()
    model.compile()
    return model

  @tf.function
  def _gradient_penalty(self, x, x_hat):
    epsilon = tf.random.uniform((self._batch_size, self._height, self._width, 1), 0.0, 1.0)
    u_hat = epsilon * x + (1 - epsilon) * x_hat
    with tf.GradientTape() as penalty_tape:
      penalty_tape.watch(u_hat)
      func = self.critic(u_hat)
    grads = penalty_tape.gradient(func, u_hat)
    norm_grads = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    regularizer = tf.math.square( tf.reduce_mean((norm_grads - 1) ) )
    return regularizer


  @tf.function
  def _get_critic_output( self, samples, fake_samples ):
    # calculate critic outputs
    real_output = self.critic(samples, **self._tf_call_kw)
    fake_output = self.critic(fake_samples, **self._tf_call_kw)
    return real_output, fake_output

  @tf.function
  def _get_critic_loss( self, samples, fake_samples, real_output, fake_output ):
    grad_regularizer_loss = tf.multiply(self._grad_weight, self._gradient_penalty(samples, fake_samples)) if self._use_gradient_penalty else 0
    critic_loss = tf.add( self.wasserstein_loss(real_output, fake_output), grad_regularizer_loss )
    return critic_loss, grad_regularizer_loss

  def _get_gen_loss( self, fake_samples, fake_output ):
    gen_loss = tf.reduce_mean(fake_output)
    return gen_loss

  def _apply_critic_update( self, critic_tape, critic_loss ):
    critic_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
    self._critic_opt.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
    return

  def _apply_gen_update( self, gen_tape, gen_loss):
    gen_grads = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
    self._gen_opt.apply_gradients(zip(gen_grads, self.generator.trainable_variables))
    return

  @tf.function
  def _train_critic(self, samples):
    with tf.GradientTape() as critic_tape:
      fake_samples = self.generate( self._batch_size )
      real_output, fake_output = self._get_critic_output( samples, fake_samples )
      critic_loss, grad_regularizer_loss = self._get_critic_loss( samples, fake_samples, real_output, fake_output)
    # critic_tape
    self._apply_critic_update( critic_tape, critic_loss )
    return critic_loss, grad_regularizer_loss

  @tf.function
  def _train_step(self, samples):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as critic_tape:
      fake_samples = self.generate( self._batch_size )
      real_output, fake_output = self._get_critic_output( samples, fake_samples )
      critic_loss, critic_regularizer = self._get_critic_loss( samples, fake_samples, real_output, fake_output)
      gen_loss = self._get_gen_loss( fake_samples, fake_output )
    # gen_tape, critic_tape
    self._apply_critic_update( critic_tape, critic_loss )
    self._apply_gen_update( gen_tape, gen_loss )
    return critic_loss, gen_loss, critic_regularizer

def preprocess_data(BASE_DIR):
  # Set some variables to preprocess the data

  HEIGHT      = 128                                                                                           # Height of the images that will be resized
  WIDTH       = 128                                                                                           # Width of the images that will be resized
  RATIO       = 8/10                                                                                          # Ratio proportion of the images on the train dataset and the validation dataset
  SIZE        = (WIDTH,HEIGHT)                                                                                # Set the size of the preprocessing resize of the images as a tuple with (HEIGHT,WIDTH)
  IMAGES_PATH = [os.path.join(BASE_DIR, filename) for filename in os.listdir(BASE_DIR) if ".png" in filename] # List with all the directory locations of the images
  NUM_IMAGES  = len(IMAGES_PATH)                                                                              # Calculate the number of images
  TRAIN_SIZE  = int(round(NUM_IMAGES*RATIO))                                                                  # Calculate the number of images in the train data set

  data_images = []                                                                                            # Set the list that will be filled with the images

  # for every image path in the list of IMAGES_PATH get an image in the directory location, resize and convert to grayscale and transform to array and append in the data_images
  for path in IMAGES_PATH:
    img  = Image.open(path).resize(SIZE).convert('L')                                                         # Convert an image to resize to SIZE(WIDTH,HEIGHT) and convert to grayscale
    data = asarray(img)                                                                                       # Convert the preproced image to an array
    data_images.append(data)                                                                                  # Append the image to an array

  data_images = np.array(data_images,dtype='f')                                                               # Convert the 8-bit image int to float
  data_images.resize((NUM_IMAGES,HEIGHT,WIDTH,1))                                                             # Resize the array with an additional dimension to convert to a tensor after
  data_images = data_images/255.0                                                                             # Normalize the image to pixels between 0 and 1

  train_data      = data_images[0:TRAIN_SIZE][:][:][:]                                                        # Split data image list into train_data with TRAIN_SIZE images
  validation_data = data_images[TRAIN_SIZE:][:][:][:]                                                         # Split data image list into validation_data with NUM_IMAGES-TRAIN_SIZE images

  return train_data,validation_data                                                                           # Return the lists with train_data and validation_data with preproced images ready to be trained in the model

# Preproccess dataset with chest x-ray images of China with the pattern and split into train and validation data, that is a list with the images witht the preset proportion
# TRAIN_IMAGES_DIR = ''
# train_data,val_data = preprocess_data(TRAIN_IMAGES_DIR)

# wass_gan = Wasserstein_GAN()        # Create an object of WGAN
# wass_gan.load('chkp_2/check_file')  # Load the weights of the models pre trained

# Train WGAN and save losses of the model
# losses = wass_gan.train(train_data_image

# Plot the graph of the model loss
def plot_graph(losses):
  disc_step_loss = losses['critic']
  gen_step_loss = losses['generator']
  reg_step_loss = losses['regularizer']
  plt.figure(figsize=(12, 5))
  plt.ylim(-30,30)
  plt.plot(gen_step_loss, label="Generator Loss")
  plt.plot(disc_step_loss, label="Discriminator Loss")
  plt.plot(reg_step_loss, label="Regularizer Loss")
  plt.grid(True, "both", "both")
  plt.legend()
  plt.savefig('graph')

def generate_new_image(gan):
  # Generate a new sample from the generator model from the latent space
  new_image = gan.generate(1)                         # Generate 1 new image
  new_image = np.array(new_image).reshape((128,128))   # Resize and convert into an array the image from the generator
  plt.imshow(new_image,cmap='gray')                    # Show and plot the image
