from pickle import load
from keras.preprocessing.text import Tokenizer
def load_doc(filename):
	file = open(filename,'r')
	text = file.read()
	file.close()
	return text

def load_training_data(filename):
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

filename = '/home/uj/Desktop/Resources/Flickr8k/Flickr8k_text/Flickr_8k.trainImages.txt'
train_data = load_training_data(filename)
print('Dataset: %d' %len(train_data) )

train_descriptions = load_descriptions('/home/uj/Desktop/Resources/descriptions.txt',
	train_data)
print('Descriptions: %d'%len(train_descriptions))

train_features = load_photo_features('/home/uj/Desktop/Resources/features.pkl',train_data)
print('Photos: %d'%len(train_features))

tokenizer = create_tokenizer(train_descriptions)
vocab_length = len(tokenizer.word_index)+1
print('Vocabulary Size: %d'%vocab_length)
