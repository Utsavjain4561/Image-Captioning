from numpy import argmax
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
def load_doc(filename):
	file = open(filename,'r')
	text = file.read()
	file.close()
	return text
def load_set(filename):
	doc = load_doc(filename)
	dataset = list()
	for line in doc.split('\n'):
		if len(line)<1:
			continue
		img_id = line.split('.')[0]
		dataset.append(img_id)
	return set(dataset)
def load_descriptions(filename,train_data):
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
		tokens = line.split()
		img_id,img_desc = tokens[0],tokens[1:]
		if img_id in train_data:
			if img_id not in descriptions:
				descriptions[img_id]=list()
			desc = 'startseq '+' '.join(img_desc)+' endseq'
			descriptions[img_id].append(desc)
	return descriptions
def load_photo_features(filename,train_data):
	all_features = load(open(filename,'rb'))
	features = {k:all_features[k] for k in train_data}
	return features
def to_lines(descriptions):
	all_descriptions=list()
	for key in descriptions.keys():
		[all_descriptions.append(d) for d in descriptions[key]]
	return all_descriptions

def create_tokenizer(descriptions):
	lines = to_lines(descriptions)
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer
	lines = to_lines(descriptions)
	return max(len(d.split()) for d in lines)
 
def max_length(descriptions):
	lines = to_lines(descriptions)
	return max(len(d.split()) for d in lines)

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

def evaluate(model,descriptions,photo,tokenizer,max_length):
	actual,predicted = list(),list()
	for key,desc_list in descriptions.items():
		y = generate_description(model,tokenizer,photo[key],max_length)
		ref = [d.split() for d in desc_list]
		actual.append(ref)
		predicted.append(y.split())

	print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
	print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
	print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
	print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))


filename = '/home/uj/Desktop/Resources/Flickr8k/Flickr8k_text/Flickr_8k.trainImages.txt'
train_data = load_set(filename)
print('Dataset: %d' %len(train_data) )

train_descriptions = load_descriptions('/home/uj/Desktop/Resources/descriptions.txt',
	train_data)
print('Descriptions: %d'%len(train_descriptions))

train_features = load_photo_features('/home/uj/Desktop/Resources/features.pkl',train_data)
print('Photos: %d'%len(train_features))

tokenizer = create_tokenizer(train_descriptions)
vocab_length = len(tokenizer.word_index)+1
print('Vocabulary Size: %d'%vocab_length)

max_length = max_length(train_descriptions)
print('Max Description length %d'%max_length)

filename = '/home/uj/Desktop/Resources/Flickr8k/Flickr8k_text/Flickr_8k.testImages.txt'
test = load_set(filename)
print('Dataset : %d' %len(test))

test_description = load_descriptions('/home/uj/Desktop/Resources/descriptions.txt',
	test)
print('Description: %d'%len(test_description))

test_features  = load_photo_features('/home/uj/Desktop/Resources/features.pkl',
	test)
print('Photos %d'%len(test_features))

filename = 'model_7.h5'
model = load_model(filename)

evaluate(model,test_description,test_features,tokenizer,max_length)