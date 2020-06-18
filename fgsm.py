from PIL import Image as Imagesave

import sys
import numpy as np

import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['figure.figsize'] = (8, 8)
mpl.rcParams['axes.grid'] = False
cls_list = ['cat', 'dog']

pretrained_model = tf.keras.models.load_model( 'model.h5')
pretrained_model.trainable = False


# Helper function to preprocess the image so that it can be inputted in MobileNetV2
def preprocess(image):
  image = tf.cast(image, tf.float32)
  image = tf.image.resize(image, (224, 224))
  image = image[None, ...]

  return image

# 讀取檔案
image_raw = tf.io.read_file('dog1.png')
image = tf.image.decode_image(image_raw)
image = preprocess(image)

image_probs = pretrained_model.predict(image)[0]
top_inds = image_probs.argsort()[::-1][:5]

# 使用Crossentropy當loss fuction
loss_object = tf.keras.losses.CategoricalCrossentropy()

def create_adversarial_pattern(input_image, input_label):
  with tf.GradientTape() as tape:
    tape.watch(input_image)
    prediction = pretrained_model(input_image)
    loss = loss_object(input_label, prediction)

  # Get the gradients of the loss w.r.t to the input image.
  gradient = tape.gradient(loss, input_image)
  # Get the sign of the gradients to create the perturbation
  signed_grad = tf.sign(gradient)
  return signed_grad

# Get the input label of the image.
# 這邊如果你的圖是dog則 tf.one_hot input 吃dog_index，這樣他才會產生變成貓的noise
# cat則使用cat_index
dog_index = 1
cat_index = 0
label = tf.one_hot(dog_index, image_probs.shape[-1])
label = tf.reshape(label, (1, image_probs.shape[-1]))
# label = image_probs

perturbations = create_adversarial_pattern(image, label)

# plt.imshow(perturbations[0]*0.5+0.5); # To change [-1, 1] to [0,1]

epsilons = [0, 0.01, 0.1, 0.15]
descriptions = [('Epsilon = {:0.3f}'.format(eps) if eps else 'Input')
                for eps in epsilons]

for i, eps in enumerate(epsilons):
  # 生產出adversaria的圖，使用epsilons來控制noise的影響
  adv_x = image + eps*perturbations
  adv_x = tf.cast(adv_x, tf.uint8)
  tf.keras.preprocessing.image.save_img('adversarial_dog1_'+str(i)+'.png',adv_x[0])

print('Finish!')