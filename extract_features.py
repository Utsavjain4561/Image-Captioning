from tqdm import tqdm
from os import listdir
from pickle import dump
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model 
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import preprocess_input
from keras.layers import BatchNormalization,Dense,RepeatVector
from keras.utils import plot_model

def extract_features(directory_name):
	# model = VGG16()
	# model.layers.pop()
	# model = Model(inputs=model.inputs,outputs=model.layers[-1].output)
	model = InceptionV3(weights='imagenet')
	model.layers.pop()
	for layer in model.layers:
		layer.trainable = False
	plot_model(model,to_file='inception.png')
	embedding_size = 300
	dense_input = BatchNormalization(axis=-1)(model.output)
	image_dense = Dense(units=embedding_size)(dense_input)
	image_embedding  = RepeatVector(1)(image_dense)
	image_input = model.input
	model = Model(inputs=model.inputs,outputs=model.layers[-1].output)

	#print(model.summary())
	features = dict()
	
	for file in tqdm(listdir(directory_name)):
		filename = directory_name+'/'+file
		image = load_img(filename,target_size=(299,299))
		image = img_to_array(image)
		image = image.reshape((1,image.shape[0],image.shape[1],
			image.shape[2]))
		image = preprocess_input(image)
		feature=model.predict(image)
		image_id = file.split('.')[0]
		features[image_id] = feature
		
	return features

directory_name = 'Resources/Flickr8k/Flickr8k_Dataset'
features = extract_features(directory_name)
print('Number of features extracted: %d' % len(features))

dump(features,open('features.pkl','wb'))