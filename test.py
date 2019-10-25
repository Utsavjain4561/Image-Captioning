from gtts import gTTS
import os
from PIL import Image 
from numpy import argmax,argsort
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

### Beam Search to predict top k captions for an image
def beam_search(image,beam_index,tokenizer,max_length,photo):
	start = [tokenizer.word_index['startseq']]
	start_word = [[start,0.0]]

	while len(start_word[0][0])<max_length:
		temp = []
		for s in start_word:
			sequence = pad_sequences([s[0]],maxlen =max_length,padding='post')
			y = model.predict([photo,sequence],verbose=0)
			yy = argsort(y[0])[-beam_index:]
			for w in yy:
				next_cap,prob = s[0][:],s[1]
				next_cap.append(w)
				prob+=y[0][w]
				temp.append([next_cap,prob])
		start_word = temp
		start_word = sorted(start_word,reverse=False,key=lambda l:l[1])
		start_word = start_word[-beam_index:]
	start_word = start_word[-1][0]
	intermedidate_caption = [word_for_id(i,tokenizer) for i in start_word]
	final_caption=[]
	for i in intermedidate_caption:
		if i!='endseq':
			final_caption.append(i)
		else:
			break
	final_caption = ' '.join(final_caption[1:])
	return final_caption


tokenizer = load(open('tokenizer.pkl','rb'))
max_length = 34
model = load_model('model_7.h5')

photo = extract_features('example.jpg')
description = generate_description(model,tokenizer,
	photo,max_length)

# speak = gTTS(text=description,lang='en')
# speak.save("audio1.mp3")
# os.system("mpg321 audio.mp3")
print(description)

try_image = 'example.jpg'
beam_description = beam_search(try_image,3,tokenizer,max_length,photo)
print('Beam Description with k=3 ',beam_description)
beam_description = beam_search(try_image,5,tokenizer,max_length,photo)
print('Beam Description with k=5 ',beam_description)
beam_description = beam_search(try_image,7,tokenizer,max_length,photo)
print('Beam Description with k=7 ',beam_description)