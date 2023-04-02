import os
from flask import Flask, request, jsonify
import joblib
import string
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from dotenv import load_dotenv

# load .env
load_dotenv()

nltk.download('stopwords')
nltk.download('punkt')

app = Flask(__name__)

# load model
model = joblib.load('classifier_v14_data3000_split80-20_ngram1-3_modeFeatureSeletionOptimizeAlphaPSO.joblib')
# load vectorizer
vectorizer = joblib.load('vectorizer_v14_data3000_split80-20_ngram1-3_modeFeatureSeletionOptimizeAlphaPSO.joblib')
# load index_feature_selected
index_feature_selected = joblib.load('ifs_v14_data3000_split80-20_ngram1-3_modeFeatureSeletionOptimizeAlphaPSO.joblib')

apiVersion = '/ml-api/v1'

@app.route(f"{apiVersion}/predict", methods=['POST'])
def predict():
    # check screet key with screet key in .env
    if request.headers['Authorization'].split(' ')[1] != os.getenv('SECRET_KEY'):
        return jsonify({
            'statusCode': 401,
            'error': 'Unauthorized',
        })

    if request.method == 'POST':
        text = request.json['text']
        cleaned_text = preprocess_text(text)
        transformed_text = vectorizer.transform([cleaned_text])
        print(transformed_text.shape)
        feature_selected = transformed_text[:, index_feature_selected]
        print(feature_selected.shape)

        result = model.predict(feature_selected)

        prob = model.predict_proba(feature_selected)[0]

        prediction = ''
        if result[0] == 0:
            prediction = 'valid'
        elif result[0] == 1:
            prediction = 'hoaks'
        else:
            prediction = 'disinformasi'

        return jsonify({
            'statusCode': 200,
            'message': 'success',
            'prediction': prediction,
            'details': {
                    'valid': '{:.2f}%'.format(prob[0] * 100),
                    'hoaks': '{:.2f}%'.format(prob[1] * 100),
                    'disinformasi': '{:.2f}%'.format(prob[2] * 100),
                }
            })
    else:
        return jsonify({
            'statusCode': 500,
            'error': 'Internal Server Error',
            })



# preprocessing
def preprocess_text(text):

    # lowercase
    text = text.lower()

    # tokenizing
    tokens = nltk.tokenize.word_tokenize(text)

    # remove punctuation
    tokens = [word for word in tokens if word not in string.punctuation]

    # remove number
    # tokens = [word for word in tokens if word.isalpha()]

    # remove stopwords
    stopword = nltk.corpus.stopwords.words('indonesian')
    tokens = [word for word in tokens if word not in stopword]

    # stemming
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    stemmed = []
    for word in tokens:
        c = word.replace('.', '').replace(',', '')
        if c.isnumeric():
            stemmed.append(word)
        else:
            stemmed.append(stemmer.stem(word))

    # join tokens menjadi text kembali
    text = ' '.join(tokens)
    
    return text



if __name__ == '__main__':
    app.run(debug=True)
