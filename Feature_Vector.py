import os
import threading
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K

from config import load_args
from data.list_generator import ListGenerator
from data.load_video import load_video_frames
from lip_model.preproc_and_aug import resize_no_crop, \
  resize_vids, random_crop_frames, \
  random_hor_flip_frames, normalize_mean_std, replicate_to_batch
from tensorflow.keras import layers
from util.tf_util import shape_list, batch_normalization_wrapper
from lip_model.resnet import resnet_18



config = load_args()


def init_data():

  print ('Loading data generators')
  val_gen = ListGenerator(data_list=config.data_list)
  val_epoch_size = val_gen.calc_nbatches_per_epoch()
  print ('Done')
  chars = val_gen.label_vectorizer.chars

  return val_epoch_size, chars, val_gen


def preprocess_and_augment(input_tens, aug_opts, flip_prob=0.5):
    output = input_tens
    
    # convert to grayscale if RGB
    if not config.img_channels == 1:
      assert config.img_channels == 3, 'Input video channels should be either 3 or 1'
      output = tf.image.rgb_to_grayscale(output)

    if config.resize_input:
      new_h = new_w = config.resize_input
      output = resize_no_crop(output,new_h, new_w)


    img_width = output.shape.as_list()[2]
    crp = img_width - config.net_input_size
    if 'crop_pixels' in aug_opts and aug_opts['crop_pixels']:
      crp -= 2 * aug_opts['crop_pixels']
    crp //= 2

    crp_l = crp_r = crp_t = crp_b = crp
    if config.scale:
      output = resize_vids(output,scale=config.scale)

    output = layers.Cropping3D(cropping=((0, 0), (crp_t, crp_b), (crp_l, crp_r)))(output)

    if 'crop_pixels' in aug_opts and aug_opts['crop_pixels']:
      output = aug_crop = random_crop_frames(output, aug_opts['crop_pixels'])

    if 'horizontal_flip' in aug_opts and aug_opts['horizontal_flip']:
      output = aug_flip = random_hor_flip_frames(output, prob=flip_prob)

    if config.mean and config.std:
      output = normalize_mean_std(output, mean=config.mean, std=config.std)

    return output


def temporal_batch_pack(input, input_shape):
  newshape = (-1,) + input_shape[1:]
  return tf.reshape(input, newshape )



def temporal_batch_unpack(input, time_dim_size, input_shape):
  newshape = (-1, time_dim_size)  + input_shape
  return tf.reshape(input, newshape)




def visual_frontend(input):

  model = input

  aug_opts = {}
  if config.test_aug_times:
    assert model.shape[0] == 1, 'Test augmentation only with bs=1'

    no_aug_input = model
    model = replicate_to_batch(model, config.test_aug_times)
    aug_opts = { 'horizontal_flip': config.horizontal_flip ,
                   'crop_pixels': config.crop_pixels,
                   }
  
    no_aug_out = preprocess_and_augment(no_aug_input, aug_opts={})
  flip_prob = 0.5 if not config.test_aug_times == 2 else 1

  aug_out = model = preprocess_and_augment(model, aug_opts = aug_opts, flip_prob=flip_prob)

  
  
  if config.test_aug_times:
    aug_out = model = tf.concat( [ no_aug_out,  aug_out], 0 )
  
  
  # spatio-temporal frontend
  model = tf.keras.layers.ZeroPadding3D(padding=(2, 3, 3))(model)
  
  model = tf.keras.layers.Conv3D(filters = 64,
                             kernel_size = (5, 7, 7),
                             strides = [1, 2, 2],
                             padding = 'valid',
                             use_bias = False)(model)
  
  model = batch_normalization_wrapper(model)
  model = tf.nn.relu(model)
  model = tf.keras.layers.ZeroPadding3D(padding=(0, 1, 1))(model)
  model = tf.keras.layers.MaxPooling3D(pool_size=(1, 3, 3), strides=(1,2,2))(model)
  
   

  packed_model = temporal_batch_pack(model, input_shape=K.int_shape(model)[1:])
  
  resnet = resnet_18(packed_model)
  

  output = temporal_batch_unpack(resnet,
                                        shape_list(model)[1],
                                        input_shape=K.int_shape(resnet)[1:])

  return output




def evaluate_model():
  val_epoch_size, chars, val_gen = init_data()

  v_idx = 0
  all_samples = np.loadtxt(config.data_list, str, delimiter=', ')

  
  
  for i in range(val_epoch_size):
    frames_batch = []
    labels_batch = []

    video_frames = []
    cnt = 0
    while cnt< config.batch_size:
        with threading.Lock():
          vid, label = all_samples[v_idx]
          v_idx += 1
        frames = load_video_frames( os.path.join(config.data_path, vid),
                                  maxlen=config.maxlen,
                                  pad_mode=config.pad_mode,
                                  grayscale=config.img_channels == 1
                                  )
        video_frames.append(frames)
        labels_batch.append(label)
        cnt+=1

    assert len(video_frames) == config.batch_size
    video_frames = np.stack(video_frames, axis = 0)
    frames_batch = [video_frames]

    for frame in frames_batch:
      output = visual_frontend(frame)
      print(output.shape)
        
        


def main():
  evaluate_model()

if __name__ == '__main__':
  main()
