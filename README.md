# Statistical-Machine-Translation
﻿		         Statistical Machine Translation(English to Hindi TextData)
------------------------------------------------------------------------------------------------------

I have worked on English to hindi parallel corpus data. I have used moses software to convert this.

First I have installed following packages in my ubuntu 14.04 
g++,git,subversion,automake,libtool,zlib1g-dev,libboost-all-dev,libbz2-dev,liblzma-dev,python-dev,libtcmalloc-minimal4

Then I have installed Boost,Moses,GIZA++,IRSTLM etc.
--------------------------------------------------------------------------------------------------------------

Dataset:
I have a dataset consisting of 14,91,827 parallel engish to hindi corpus dataset. It is kept in my IITPatna folder(under rajeev_intern directory).
I have made a training model of this corpus dataset using run decoder through moses software.
I have  got the blue score value 46.58 out of 100.
Now I have used Neural approach to find stack size and beam value of this tested data(test.en.text)
--------------------------------------------------------------------------------------------------------------------
Neural Approach:-I have calculated in two ways.
-----------------------------------------------------------
1. Through regression on four parameters 
	a. Percentage of comma
	b. Percentage of long sentences 
	c. Average words per line in a text
	d. Percentage of Stop words
Model Used:- Sequential Model(Feed Forward Neural Network)
Activation:- Softmax
Optimiser:- Adam
epochs:- 10
Mean Square Error:-0.61(stack_data.csv)
Mean Square Error:-0.43(beam_data.csv)
I have dataset based on these parameters(beam_data.csv && stack_data.csv) which is saved under(rajeev_intern/set_eng_to_hindi/test/regression)
I have used sequential model and calculated mean square error. I got mean square error to be approx 0.6 which is quite good.
Then I have used (test.text) file for training. First I have calculated all the four parameters in this file through python code.
	for calculating Percentage of comma(comma_percentage.py)
	for calculating Percentage of long sentences(long_line_percentage.py)
	for calculating Average words(avg_words_per_line.py)
	for calculating Percentage of Stop Words(percentage_of_stop_word.py)
After that I have calculated the value and then through moses software I have given the value of b and s and I got blue score value to be 46.58(rajeev_intern/set_eng_to_hindi/test/regression/bleu_score)
----------------------------------------------------------------------------------------------------------------------------------------------------
2. Through word-vec embedding through glove in python 3.6
Model used:-Sequential
Sequence Processor:- This is a word embedding layer for handling the text input, followed by a Long Short-Term Memory (LSTM) recurrent neural network layer.
Activation:- Softmax
Optimiser:- Adam
epochs:- 10
metrics:- accuracy
loss:- categorical_crossentropy
I have a dataset of 51 files(parellel english-hindi files), each file containing average of 3000 words. I have find max word and min word using python code.
I have dataset based on corresponding 51 files(beam_data.csv && stack_data.csv) which is saved under(rajeev_intern/set_eng_to_hindi/test/word_vect)
I have used glove (embedding length-300) for each word. I have kept 2000 LSTM for each word in the file.
I have used cuDNNLSTM instead of simple LSTM as cuDNN makes the processing as it uses the external graphics so there in not much load cpu processor.
I have made the dataset more cleaner by removing extra spaces(remove_linebreak.py) through python code.
I have calculated loss using categorical crossentropy.
I run the code  on my system but my system fails because of large number of LSTM used.
So I have run the code on gpu(on Spyder(Anaconda3)).
And I got the result and I have tested (test.text) this file through it.
And finally calculated bleu score value and I got 46.92(rajeev_intern/set_eng_to_hindi/test/word_vect/bleu_score) which is quite good as comparison to other.
--------------------------------------------------------------------------------------------------------------------------------------------------------
I have calculate various times such as CPU time,SYS time,USER time,REAL time for each approach.
--------------------------------------------------------------------------------------------------
1.Through Moses directly
CPU  :- 34.872
SYS  :- 0.736
USER :- 34.136
REAL :- 36.108
###Bleu Score :- 46.58
-----------------------------------------------
2.Through regression of step1
CPU  :- 35.028
SYS  :- 0.484
USER :- 34.344
REAL :- 35.021
###Bleu Score :- 46.58
----------------------------------------------
3.word-vec using glove
CPU  :- 34.928
SYS  :- 0.516
USER :- 34.412
REAL :- 34.921
###Bleu Score :- 46.92
---------------------------------------------------
I have to find analysis among various sentences gievn by input by me to check the accuracy.
I have a folder consisting of each input files and corresponding output file
and we have analysed between the output and real output....
--------------------------------------------------------------------------------------------
## Requirements:-
--------------------------------------------------------------------------------------------
- Python SciPy environment installed, ideally with Python 3.
- TensorFlow or Theano backend
- Keras (2.1.5 or higher)
- numpy
- h5py
- pandas
- Pillow
- graphviz
---------------------------------------------------------------------------------------------------
Thanks
Regard
Rajeev Raj
NIT Patna
Mob:-7301010798,7301659953
