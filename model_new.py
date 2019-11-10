from keras.applications.inception_v3 import InceptionV3
from keras.layers.normalization import BatchNormalization
from keras.layers import (Dense,Dropout,Embedding,Input,LSTM,
	RepeatVector,TimeDistributed)
from keras.layers.merge import Concatenate
from keras.optimizers import Adam
from keras.models import Model
from keras.utils import plot_model

class ImageCaptionModel(object):

	def __init__(self,
		learning_rate=None,
		vocab_size=None,
		embedding_size=None,
		rnn_output_size=None,
		rnn_layers=None,
		max_length=None,
		dropout_rate=None):
		self._learning_rate = 0.00051
		self._vocab_size = 7579
		self._embedding_size = 300
		self._rnn_output_size=300
		self._rnn_layers=3
		self._max_length=34
		self._dropout_rate=0.5


	
	def build(self):

		image_input,image_embedding = self._build_image_embedding()
		sentence_input,word_embedding = self._build_word_embedding()
		sequence_input = Concatenate(axis=1)([image_embedding,
			word_embedding])
		sequence_output = self._build_sequence_model(sequence_input)

		decoder = Dense(self._embedding_size,activation='relu')(sequence_output)
		outputs = Dense(self._vocab_size,activation='softmax')(decoder)

		model = Model(inputs=[image_input,sentence_input],outputs=outputs)
		model.compile(loss='categorical_crossentropy',optimizer='adam')
		print(model.summary())
		plot_model(model,to_file='model.png')
		return model

	def _build_image_embedding(self):
		# model = InceptionV3(weights='imagenet',pooling='avg')
		# model.layers.pop()
		# for layer in model.layers:
		# 	layer.trainable = False

			input_layer = Input(shape=(2048,))
			dense_input = BatchNormalization(axis=-1)(input_layer)
			pure_input = Dropout(self._dropout_rate)(dense_input)
			image_dense = Dense(units=self._embedding_size)(pure_input)

			image_embedding = RepeatVector(1)(image_dense)
			image_input = input_layer
			return image_input,image_embedding


	def _build_word_embedding(self):

		sentence_input = Input(shape=[self._max_length])
		word_embedding = Embedding(input_dim=self._vocab_size,
			output_dim=self._embedding_size)(sentence_input)
		
		return sentence_input,word_embedding

	def _build_sequence_model(self,sequence_input):
		# input_ = sequence_input
		# for _ in range(self._rnn_layers):
		# 	input_ = BatchNormalization(axis=-1)(input_)
		# 	print(input_.shape())
		# 	rnn_out = LSTM(self._embedding_size)(input_)
		# 	input_ = rnn_out
		# time_dense = TimeDistributed(Dense(units=self._vocab_size
		# 	))(rnn_out)
		# return time_dense

		sequence_input = BatchNormalization(axis=-1)(sequence_input)
		pure_embedding = Dropout(0.5)(sequence_input)
		lstm_out = LSTM(units=self._rnn_output_size,
				dropout=self._dropout_rate)(pure_embedding)

		# sequence_output = TimeDistributed(Dense(
		# 	units=self._vocab_size))(lstm_out)
		return lstm_out

model = ImageCaptionModel()
model.build()




