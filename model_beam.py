import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.layers import CuDNNLSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn import preprocessing

np.random.seed(0)

X1,Y=[],[]
with open("new_train1.txt",'r',encoding="utf-8") as f:
	for i in f:
		if i=='\n':
			continue
		#X.append(i.strip())
		X1.append(i.strip().split())
X=[]
X1=pad_sequences(X1,dtype=str,maxlen=20000,padding="post",value=1)

for i in range(len(X1)):
	str1=" "
	for j in X1[i]:
		if j=="1.0":
			str1+=" NAW"
		else:
			str1+=" "+j
	X.append(str1.strip())
	
test=[]
new_words=[]
with open('test.txt','r' ,encoding='utf-8') as f:
	for i in f:
		test.append(i.strip())
		for j in i.strip().split():	
			new_words.append(j)
#print(new_words)
#print(test)





all_words = set(w for words in X for w in words.split())
glove= {}
total_words=[]
with open("glove.840B.300d.txt", "rb") as infile:
    for line in infile:
        parts = line.split()
        word = parts[0].decode("utf-8")
        if (word in all_words):
            nums=np.array(parts[1:], dtype=np.float32)
            total_words.append(word)
            glove[word] = nums
    total_words.append("NAW")
    words_np=list(all_words-set(total_words))
    glove["NAW"]=np.zeros((300,1))
    #print(words_np)
    for word in words_np:
    	glove[word]=np.zeros((300,1))

f=pd.read_csv('beam_data.csv',header=None)
y=list(f[4])
y=y[1:]
lb = preprocessing.LabelBinarizer()
lb.fit(y)
Y=lb.transform(y)
#print(len(Y))
#print(len(y))

tokenizer = Tokenizer(num_words=43400)
tokenizer.fit_on_texts(list(all_words)+new_words)
X=tokenizer.texts_to_sequences(X)

X1=np.empty((0,200),dtype=np.float32)
for i in X:
    i=np.array(i)
    i=i[:200].reshape((1,200))
    X1=np.append(X1,i,axis=0)
    
print(X1.shape)
#print(len(X))
#X=pad_sequences(X,maxlen=150,padding="post",value=)

test_data=tokenizer.texts_to_sequences(test[0])
#print(test_data)
 
	
	
X_train, X_test, y_train, y_test = train_test_split(X1,Y, test_size=0.3, shuffle=False)

embedding_matrix = np.zeros((len(all_words)+1, 300))
for word, i in tokenizer.word_index.items():
    embedding_vector = glove.get(word)
    if embedding_vector is not None:
        embedding_matrix[i,:] = list(embedding_vector)

model = Sequential()
model.add(Embedding(len(all_words)+1,300,weights=[embedding_matrix],input_length=200,trainable=False))
model.add(CuDNNLSTM(200)) # return sequence=false #relu/leakyrelu
model.add(Dense(3)) #time distributed
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) #mse
print(model.summary())
model.fit(X_train, y_train, epochs=10, batch_size=16)
 

preds=model.predict(X_test)
print(np.argmax(preds,axis=1)[0])
#print(np.argmax(y_test,axis=1)[0])



		
