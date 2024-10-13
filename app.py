from flask import Flask, request, render_template
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = Flask(__name__)

# Load the saved model and tokenizer
model_path = './emotion_model'
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

# Define emotion labels
emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    result = None
    if request.method == 'POST':
        text = request.form['text']
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
        outputs = model(**inputs)
        prediction = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = prediction.argmax().item()
        predicted_emotion = emotion_labels[predicted_class]
        confidence = prediction[0][predicted_class].item()
        result = {
            'text': text,
            'emotion': predicted_emotion,
            'confidence': confidence
        }
    return render_template('predict.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
