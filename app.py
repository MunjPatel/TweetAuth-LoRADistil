from flask import Flask, render_template, request, redirect, url_for
import torch
import numpy as np
from transformers import DistilBertTokenizer, AutoModelForSequenceClassification
from lora_layers import LinearWithLoRA, LoRALayer, apply_lora_to_model
import os
import __main__

setattr(__main__, "LinearWithLoRA", LinearWithLoRA)
setattr(__main__, "LoRALayer", LoRALayer)

app = Flask(__name__)

# Load tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)

# Attempt to load model directly if it's a full model
model_path = os.path.join(os.getcwd(), "best_model.pth")
try:
    model = torch.load(model_path, map_location=torch.device("cpu"))
except TypeError:
    # If loading directly fails, fall back to loading as `state_dict`
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    apply_lora_to_model(model)
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict, strict=False)

model.eval()  # Set the model to evaluation mode

# Function to predict label and probability for a given tweet
def predict_tweet(tweet_text):
    inputs = tokenizer.encode_plus(
        tweet_text,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # Perform inference
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    probs = torch.softmax(logits, dim=1).flatten().cpu().numpy()
    predicted_label = np.argmax(probs)
    probability = probs[predicted_label]

    label = "Real" if predicted_label == 1 else "Fake"
    return label, probability

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        tweet_text = request.form['tweet_text']
        label, probability = predict_tweet(tweet_text)
        return render_template('result.html', label=label, probability=probability)

@app.route('/back')
def back():
    return redirect(url_for('home'))

if __name__ == "__main__":
    app.run()
