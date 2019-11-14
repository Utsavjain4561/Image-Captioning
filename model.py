from keras.models import Model 
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.utils import plot_model
import os
os.environ["PATH"] += os.pathsep + '/home/uj/graphviz_2.38.0-12ubuntu2_amd64/usr/bin'
embeddingsize = 300
features = 2048
def caption_model(vocab_length,max_length):

	input1 = Input(shape=(features,))
	layer1 = Dropout(0.5)(input1)
	layer2 = Dense(embeddingsize,activation='relu')(layer1)

	input2 = Input(shape=(max_length,))
	layer3 = Embedding(vocab_length,embeddingsize,mask_zero=True)(input2)
	layer4 = Dropout(0.5)(layer3)
	layer5 = LSTM(embeddingsize)(layer4)

	decoder1 = add([layer2,layer5])
	decoder2 = Dense(embeddingsize,activation='relu')(decoder1)
	outputs = Dense(vocab_length,activation='softmax')(decoder2)

	model = Model(inputs=[input1,input2],outputs=outputs)
	model.compile(loss='categorical_crossentropy',optimizer='adam')

	#print(model.summary())
	#plot_model(model,to_file='model.png',show_shapes=True)
	return model
#caption_model(8763,34)

