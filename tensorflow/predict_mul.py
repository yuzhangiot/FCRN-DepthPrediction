import argparse
import os, sys
from os import listdir
from os.path import isfile, join
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image

import models

def predict(model_data_path, image_root, output_root):

    
    # Default input size
    height = 228
    width = 304
    channels = 3
    batch_size = 1
   
   
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
	
   	fileNum = len([name for name in os.listdir(image_root)])
   	print(image_root)
   	print(fileNum)
   	for idx in range(0, fileNum):
	    image_path = image_root + "color_out" + str(idx+1).zfill(6) + ".png"
	    output_path = output_root + "depth_out" + str(idx+1).zfill(6) + ".png"
	    # Read image
	    img = Image.open(image_path)
	    img = img.resize([width,height], Image.ANTIALIAS)
	    img = np.array(img).astype('float32')
	    img = np.expand_dims(np.asarray(img), axis = 0)

       	    # Evalute the network for the given image
            pred = sess.run(net.get_output(), feed_dict={input_node: img})
        
            # Plot result
            fig = plt.figure()
            ii = plt.imshow(pred[0,:,:,0], interpolation='nearest')
            #fig.colorbar(ii)
	    ii.set_cmap('gray')
	    plt.axis('off')
            plt.savefig(output_path)
	    #plt.show()
        
                
def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Converted parameters for the model')
    parser.add_argument('image_paths', help='Directory of images to predict')
    parser.add_argument('output_paths', help='Directory of output images')
    args = parser.parse_args()

    # Predict the image
    img_root = args.image_paths + "/"
    output_root = args.output_paths + "/"
    pred = predict(args.model_path, img_root, output_root)
    
    os._exit(0)

if __name__ == '__main__':
    main()

        



