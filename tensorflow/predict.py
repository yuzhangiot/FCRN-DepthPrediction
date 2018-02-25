import argparse
#import cv2.cv
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
import numpngw
import scipy.ndimage

import models

np.set_printoptions(threshold=np.nan)
#print(cv.__version__)

def predict(model_data_path, image_path):

    
    # Default input size
    height = 228
    width = 304
    channels = 3
    batch_size = 1
   
    # Read image
    img = Image.open(image_path)
    img = img.resize([width,height], Image.ANTIALIAS)
    img = np.array(img).astype('float32')
    img = np.expand_dims(np.asarray(img), axis = 0)
   
    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))

    # Construct the network
    net = models.ResNet50UpProj({'data': input_node}, batch_size, 1, False)
        
    with tf.Session() as sess:

        # Load the converted parameters
        print('Loading the model')

        # Use to load from ckpt file
        saver = tf.train.Saver()     
        saver.restore(sess, model_data_path)

        # Use to load from npy file
        #net.load(model_data_path, sess) 

        # Evalute the network for the given image
        pred = sess.run(net.get_output(), feed_dict={input_node: img})
        # print(pred.shape)
        # print(pred[0,:,:,0])
        
        
        pre_new = pred[0,:,:,0]
        # pre_new = pre_new * 1000 * 5

        pre_resize = scipy.ndimage.zoom(pre_new, 4, order=0)
        pre_resize = pre_resize * 1000 * 5

        img = np.zeros((128 * 4, 160 * 4, 1), dtype=np.uint16)
        img[:,:,0] = pre_resize
        
        # print(img)
        # print(img.shape)
        print("max is: " + str(np.amax(pre_resize)))
        print("min is: " + str(np.amin(pre_resize)))

        numpngw.write_png("test.png", img)



        # # Plot result
        # fig = plt.figure(frameon=False, figsize=(5, 4), dpi=120) #5:4,120
        # ax = fig.add_axes([0, 0, 1, 1])
        # ii = ax.imshow(pre_new,cmap=plt.cm.gray,interpolation='nearest')
        # #ii.set_cmap('gray')
        # #fig.colorbar(ii)
        # ax.axis('off')
        # plt.savefig("test.png")
        # plt.show()
        return pred
        
                
def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Converted parameters for the model')
    parser.add_argument('image_paths', help='Directory of images to predict')
    args = parser.parse_args()

    # Predict the image
    pred = predict(args.model_path, args.image_paths)
    
    os._exit(0)

if __name__ == '__main__':
    main()

        



