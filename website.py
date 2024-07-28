#!/usr/bin/env python
# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os
from pyngrok import ngrok, conf
import torch
from sbvqa2.base_model import InferenceModel
import pickle
import string
import random
from visualize import visualize_gradcam, visualize_attention

def id_generator(size=20, chars=string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

answers_list = pickle.load(open('models/trainval_label2ans.pkl', 'rb'))

@torch.no_grad()
def predict(image_path, audio_path):
    model.eval()
    images = model.load_images([image_path])
    signals_with_lens = model.load_audios([audio_path])
    answers_ids = model((images, signals_with_lens)).argmax(-1)
    result = [answers_list[i] for i in answers_ids.tolist()]
    return result

# Loading the model
batch_size = 32
num_hid = 1024
q_dim = 512
v_dim = 1024
num_classes = 3129
speech_encoder_path = 'models/stt_en_conformer_ctc_large_24500_hours_bpe.nemo'
sbvqa2_model_path = 'models/best_sbvqa_2.0_model.pt'
model = InferenceModel(q_dim, v_dim, num_hid, num_classes, speech_encoder_path,
                  rnn_type='GRU', bidirect=False, rnn_layers=1,
                  rnn_dropout=0.0, ans_gen_dropout=0.5).cuda()
model_state_dict = torch.load(sbvqa2_model_path)
model.load_state_dict(model_state_dict)
model.eval()

# --------------------------------------------------------------

# Running the server
print("Enter your authtoken, which can be copied from https://dashboard.ngrok.com/get-started/your-authtoken")
conf.get_default().auth_token = 'YOUR_TOKEN_GOES_HERE' # getpass.getpass()

app = Flask(__name__, template_folder='./', static_folder='./')

# Open a ngrok tunnel to the HTTP server
public_url = ngrok.connect(5000).public_url
print(" * ngrok tunnel \"{}\" -> \"http://127.0.0.1:{}/\"".format(public_url, 5000))

# Update any base URLs to use the public ngrok URL
app.config["BASE_URL"] = public_url

# ... Update inbound traffic via APIs to use the public-facing ngrok URL


# app = Flask(__name__)
CORS(app)  # This allows cross-origin requests, which is necessary for local development

@app.route("/", methods=['GET'])
def index():
    return render_template("index.html")

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files or 'audio' not in request.files:
        return jsonify({'error': 'Missing image or audio file'}), 400

    image = request.files['image']
    filename = id_generator()
    image_path = f'uploads/{filename}.jpg'
    with open(image_path, 'wb') as f1:
        image.save(f1)
    print('file uploaded successfully')
    audio = request.files['audio']
    filename = id_generator()
    audio_path = f'uploads/{filename}.wav'
    with open(audio_path, 'wb') as f1:
        audio.save(f1)

    result = predict(image_path, audio_path)
    gradcam_viz_img_path = visualize_gradcam(model, image_path, audio_path)
    all_heads_img_path = visualize_attention(model, image_path)
    output = {
                'answer': result[0],
                'gradcam_image_url': f'/{gradcam_viz_img_path}',
                'all_heads_image_url': f'/{all_heads_img_path}'
            }

    return jsonify(output)


@app.route('/uploads/<path:path>')
def send_report(path):
    return send_from_directory('./uploads', path)


if __name__ == '__main__':
    app.run(use_reloader=False)

