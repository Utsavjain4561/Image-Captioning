import string 
def load_file(filename):
	file = open(filename,'r')
	text = file.read()
	file.close()
	return text

filename='Resources/Flickr8k/Flickr8k_text/Flickr8k.token.txt'
doc = load_file(filename)

def load_descriptions(document):
	mapping = dict()
	for line in document.split('\n'):
		tokens = line.split()
		if len(line)<2:
			continue
		image_id,image_description = tokens[0],tokens[1:]
		image_id = image_id.split('.')[0]
		image_description=' '.join(image_description)
		if image_id not in mapping:
			mapping[image_id]=list()
		mapping[image_id].append(image_description)

	return mapping
descriptions = load_descriptions(doc)
print('Loaded: %d '%len(descriptions))

def clean_descriptions(descriptions):
	table = str.maketrans('', '', string.punctuation)
	for key, desc_list in descriptions.items():
		for i in range(len(desc_list)):
			desc = desc_list[i]
			desc = desc.split()
			# convert to lower case
			desc = [word.lower() for word in desc]
			# remove punctuation from each token
			desc = [w.translate(table) for w in desc]
			# remove hanging 's' and 'a'
			desc = [word for word in desc if len(word)>1]
			# remove tokens with numbers in them
			desc = [word for word in desc if word.isalpha()]
			# store as string
			desc_list[i] =  ' '.join(desc)
 
# clean descriptions
clean_descriptions(descriptions)

def to_vocabulary(descriptions):
	all_desc = set()
	for key in descriptions.keys():
		[all_desc.update(d.split()) for d in descriptions[key]]
	return all_desc
vocabulary = to_vocabulary(descriptions)
print('Vocalbulary Sze: %d '%len(vocabulary))


def save_desriptions(descriptions,filename):
	lines = list()
	for key,desc_list in descriptions.items():
		for desc in desc_list:
			lines.append(key+' '+desc)
	data='\n'.join(lines)
	file=open(filename,'w')
	file.write(data)
	file.close()
save_desriptions(descriptions,'descriptions.txt')



