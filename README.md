# Sentence Completion
 In this project, I try to generate meaningful text that completes a given sentence. I achieve this by using LSTMs.

The model has been trained on a novel titled Cave Girl. The model will try to learn the patterns of words in sentences found in this novel and will try to complete sentences based on this same vocabulary. The sentence completion will be done by seeding a sample sentence to the model, then the model will generate a sentence in accordance with provided sample. 

Run model.py to straight away get the results.

The model will be provided with a random incomplete sentence of about 5 words or less that exits in Cave Girl novel. The model will then attempt to complete the sentence by predicting, the next word continuosly until it predicts a terminator such as a question mark and a full-stop (".").

The model has been trained in the model_builer.py file and has been saved in JSON format as model.json. 

