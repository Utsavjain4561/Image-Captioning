from pickle import load
from numpy import array
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
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
	lines = to_lines(descriptions)
	return max(len(d.split()) for d in lines)
 
def max_length(descriptions):
	lines = to_lines(descriptions)
	return max(len(d.split()) for d in lines)
def create_sequences(tokenizer, max_length, desc_list, photo, vocab_size):
	X1, X2, y = list(), list(), list()
	# walk through each description for the image
	for desc in desc_list:
		# encode the sequence
		seq = tokenizer.texts_to_sequences([desc])[0]
		# split one sequence into multiple X,y pairs
		for i in range(1, len(seq)):
			# split into input and output pair
			in_seq, out_seq = seq[:i], seq[i]
			# pad input sequence
			in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
			# encode output sequence
			out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
			# store
			X1.append(photo)
			X2.append(in_seq)
			y.append(out_seq)
	return array(X1), array(X2), array(y)
def data_generator(descriptions, photos, tokenizer, max_length, vocab_size):
	# loop for ever over images
	while 1:
		for key, desc_list in descriptions.items():
			# retrieve the photo feature
			photo = photos[key][0]
			in_img, in_seq, out_word = create_sequences(tokenizer, max_length, desc_list, photo, vocab_size)
			yield [[in_img, in_seq], out_word]

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

max_length = max_length(train_descriptions)
print('Max Description length %d'%max_length)
generator = data_generator(train_descriptions, train_features, tokenizer, 
	max_length, vocab_length)
inputs, outputs = next(generator)
print(inputs[0].shape)
print(inputs[1].shape)
print(outputs.shape)
