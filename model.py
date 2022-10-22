from keras.models import model_from_json
import numpy as np
import random 
import sys

#Get necessary data
#Load text
filename = "cavegirl.txt"
raw_text = open(filename,'r',encoding='utf-8').read()
raw_text = raw_text.lower()
#print(raw_text[0:1000])

#clean text
#remove numbers
raw_text = "".join(c for c in raw_text if not c.isdigit())

#seperate words
words = []
temp = ''
for i in raw_text[:50000]:
    if i.isalpha():
        temp+=i
        continue
    if i.isspace():
        words.append(temp)
        temp = ''
        #words.append(i)
        continue
    words.append(temp)
    temp = ''
    words.append(i)  

#what are the unique wors and characters
unique_words = sorted(list(set(words)))

#encode the characters to numbers
#each char will be assigned a unique int
#create a dict for chars mapped to ints
word_to_int = dict((c,i) for i, c in enumerate(unique_words))

#do the reverse to map ints to chars
int_to_word = dict((i,c) for i, c in enumerate(unique_words))

#summarize the data
n_words = len(words)
n_vocab = len(unique_words)
seq_length = 5 

def sample(preds):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    #print(exp_preds,np.sum(exp_preds))
    preds = exp_preds/(np.sum(exp_preds) + 0.00000000001)
    probas = np.random.multinomial(1,preds,1)
    return np.argmax(probas)

#Load the model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

#load the saved weights
filename = 'saved_weights_cavegirl_deep2_wordtoken0space_5sl_10epochs.h5'
model.load_weights(filename)

start_index = random.randint(0,n_words - seq_length - 1)

generated = ''
sentence = words[start_index : start_index+seq_length]
generated.join(sentence)

print(len(sentence))
print("SEED: ",(" ".join(sentence))," --> ")

for i in range(70):
    x_pred = np.zeros((1,seq_length,n_vocab))
    for t, char in enumerate(sentence):
        x_pred[0,t,word_to_int[char]] = 1
        
    preds = model.predict(x_pred, verbose=0)[0]
    next_index = sample(preds)
    next_char = int_to_word[next_index]
    
    if(next_char == sentence[-1]):
        continue
    generated +=" "+next_char

    if sentence[-1].isalpha() and next_char.isalpha():
        sys.stdout.write(" "+next_char)
    else:
        sys.stdout.write(next_char)    
    sentence = sentence[1:] + [next_char]
    sys.stdout.flush()
    if(i>=5 and (next_char in [";",".",":","?","!","\n"])):
        break
print()
