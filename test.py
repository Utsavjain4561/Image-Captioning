
from numpy import argmax
from pickle import load
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
from keras.models import load_model
def extract_features(filename):

	model = VGG16()
	model.layers.pop()
	model = Model(inputs=model.inputs,
		outputs=model.layers[-1].output)

	img = load_img(filename,target_size=(224,224))
	img = img_to_array(img)

	img = img.reshape((1,img.shape[0],img.shape[1],img.shape[2]))
	img = preprocess_input(img)

	feature = model.predict(img,verbose=0)
	return feature
def word_for_id(integer,tokenizer):
	for word,index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

def generate_description(model,tokenizer,photo,max_length):

	input_text = 'startseq'
	for i in range(max_length):
		sequence = tokenizer.texts_to_sequences([input_text])[0]
		sequence = pad_sequences([sequence],maxlen=max_length)

		y = model.predict([photo,sequence],verbose=0)
		y = argmax(y)
		word = word_for_id(y,tokenizer)

		if word is None:
			break
		input_text+=' '+word
		if word == 'endseq':
			break
	return input_text

tokenizer = load(open('tokenizer.pkl','rb'))
max_length = 34
model = load_model('model_7.h5')

photo = extract_features('example.jpg')
description = generate_description(model,tokenizer,
	photo,max_length)
print(description)

