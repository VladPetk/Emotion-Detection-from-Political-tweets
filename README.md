# Emotion Detection from Political Tweets

 This is a repository for a (part of a) research project on determining the effects of negative negative campaigning on the electorate's emotional states and their subsequent voting behavior. This part of the project concerns detection of emotions (anger and fear) in the tweets that were posted as direct replies to election candidates' campaign messages during the UK 2019 parliamentary elections. 

Two different approaches are utilized - both for exploratory purposes and to achieve satisfactory results - in detecting the two emotions. For detection of fear, an LSTM architecture was used, whereas with anger I employed transfer learning and leveraged the power of Google's T5 model. The results were quite satisfactory with F1 scores averaging at 0.81. 

## Fear
For fear, a bidirectional LSTM network was built in Keras to learn to predict fear using a manually annotated dataset (N=3000). I used pre-trained word embeddings from SpaCy and engineered a number of extra-textual features (number of characters, number of pronouns, etc.) that were supplied to the model in addition to the (converted) text input. The full code can be found in the *detecting_fear_LSTM.ipynb* Jupyter Notebook.

## Anger
With anger, I decided to test out the abilities of the (ever more popular) transfer learning models. In this case, I utilized Google's T5 text-to-text transfer transformer (https://github.com/google-research/text-to-text-transfer-transformer). I fine-tuned it on manually annotated tweets (N=1000) which were transformed into TF records and stored in a GCP bucket allowing for them to be loaded directly into TPU nodes. The results were quite impressive (F1 score of 0.79 for presence of anger and 0.86 for its absence). The Jupyter Notebook (*detecting_anger_T5.ipynb*) contains the full code, including a chunk for conversion of CSV data into TF records.
