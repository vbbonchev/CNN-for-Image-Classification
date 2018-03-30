mport tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse


# find the path to the image
dir_path = os.path.dirname(os.path.realpath(__file__))
image_path=sys.argv[1] 
filename = dir_path +'/' +image_path
#knobs for adjusting the image size
image_size=128
num_channels=3
images = []
# start using openCV to read images
image = cv2.imread(filename)
# resizing the image
image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
images.append(image)
images = np.array(images, dtype=np.uint8)
images = images.astype('float32')
images = np.multiply(images, 1.0/255.0) 
#reshaping to fit the network
x_batch = images.reshape(1, image_size,image_size,num_channels)

#restoring the model
sess = tf.Session()
# recreate the graph first
saver = tf.train.import_meta_graph('dogs-cats-model.meta')
# load the weights for the graph
saver.restore(sess, tf.train.latest_checkpoint('./'))

# getting the graph
graph = tf.get_default_graph()

#y_pred is the name of the output prediction
y_pred = graph.get_tensor_by_name("y_pred:0")

#feeding the image
x= graph.get_tensor_by_name("x:0") 
y_true = graph.get_tensor_by_name("y_true:0") 
y_test_images = np.zeros((1, 2)) 



feed_dict_testing = {x: x_batch, y_true: y_test_images}
result=sess.run(y_pred, feed_dict=feed_dict_testing)
# show result in format [prob of car, prob of bike]
print(result)
