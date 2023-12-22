
#tools/download.sh
#tools/process.sh
python main.py --output /content/my_model/ \
        --train-img-feats-folder /content/blip_feats_train2014 \
        --val-img-feats-folder /content/blip_feats_val2014 \
        --audio-feats-folder /content/new_corrected_speech_features \
        --dataroot /content/bottom_up_attention_SBVQAv14_blip_feats_single_vector_v2/data