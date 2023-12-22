
unzip data.zip

# download NeMo Conformer Checkpoint (.nemo)
gdown -c --fuzzy -O pretrained_models/ https://drive.google.com/file/d/1-FD-pLvCSy_TZ7POQap_XzpzsuvDnIBv/view?usp=sharing

# BLIP precomputed features (train2014 images)
gdown -c --fuzzy -O data/ https://drive.google.com/file/d/1-AR0Krjip2SYaKWY6dQvAhamVo91pUiJ/view?usp=sharing

# BLIP precomputed features (val2014 images)
gdown -c --fuzzy -O data/ https://drive.google.com/file/d/1-Q3dDlRue9dbDV3qwbGDaF6GNLm4rN9U/view?usp=sharing

# precomputed speech features (using NeMo Conformer checkpoint)
gdown -c --fuzzy -O data/ https://drive.google.com/file/d/1Icdcw4rYyTzm4X3osAKNrsTXkyEHuMEq/view?usp=sharing

echo "unzipping new_my_blip_features_i_train2014_img_size_480_v6_large_82782.zip...."
unzip -q data/new_my_blip_features_i_train2014_img_size_480_v6_large_82782.zip -d data/
mv data/my_blip_features/ data/blip_feats_train2014

echo "unzipping new_my_blip_features_i_val2014_img_size_480_v6_large_40503.zip...."
unzip -q data/new_my_blip_features_i_val2014_img_size_480_v6_large_40503.zip -d data/
mv data/my_blip_features/ data/blip_feats_val2014

echo "unzipping all_SBVQA_features_npz_files.zip...."
unzip -q data/all_SBVQA_features_npz_files.zip -d data/
mv data/new_corrected_speech_features/ data/speech_features
