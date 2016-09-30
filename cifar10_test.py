import numpy as np
import scipy.misc
from keras.models import model_from_json

# load some image to test
def load_and_scale_imgs():
   img_names = ['data/cat.jpg', 'data/dog.jpg']
   # resize img to 32x32 outs (32, 32, 3) dim, then transpose to make it (3, 32, 32)
   imgs = [np.transpose(scipy.misc.imresize(scipy.misc.imread(img_name), (32, 32)), (2, 0, 1)).astype('float32') for img_name in img_names]
   # combine the list of image tensors into a single tensor and normalize between 0-1
   return np.array(imgs) / 255

def load_model(model_def_fname, model_weight_fname):
   model = model_from_json(open(model_def_fname).read())
   model.load_weights(model_weight_fname)

   return model

classes = ['airplane', 'automobile', 'bird', 'cat', 'dog', 'frog', 'horse', 'ship', 'truck']

if __name__ == '__main__':
   imgs = load_and_scale_imgs()
   model = load_model('data/cifar10_architecture.json', 'data/cifar10_weights.h5')
   predictions = model.predict_classes(imgs)
   for i in predictions:
       print(classes[i])
