# SBVQA 2.0 Official Implementation
This is the official implementation of our paper:
> [SBVQA 2.0: Robust End-to-End Speech-Based Visual Question Answering for Open-Ended Questions](https://ieeexplore.ieee.org/document/10343139)


## How to run?
1. Install all requirements in `requirements.txt`.
2. Download models inside the folder `models/`.
3. Make a new empty folder with the name `uploads/`.
4. Create an account in [ngrok](https://ngrok.com) to be able to use the model from browser via internet.
5. Copy your [ngrok authtoken](https://dashboard.ngrok.com/get-started/your-authtoken).
6. Edit `website.py` script by replacing `YOUR_TOKEN_GOES_HERE` string with your `ngrok authtoken`.
7. Run the script `python website.py`
8. Enjoy ;)

## Data
### Audio files

**`SBVQA 2.0 dataset = SBVQA 1.0 dataset + The complementary spoken questions`**

- SBVQA 1.0 original data (identical copy of the data from [zted/sbvqa](https://github.com/zted/sbvqa) repo): [Download](https://drive.google.com/file/d/1-DzJbt5jwXGeRnvgTw2gTm4fC8PtLhCR/view)
- The complementary spoken questions: [Download](https://drive.google.com/file/d/1_YNontdvxKmF92AYW8XxSpLv37rst4OR/view)

Also, you can download `mp3_files_by_question.pkl`, a mapper where the key is the textual question and the value is the `.mp3` file name, from [this link](https://drive.google.com/file/d/1HtVK15wjj2MzQM5ApouZ6Kp-305eGptu/view).

To load the mapper, use the following code snippet:
```python
import re
import pickle

def clean_question(text):
    text = text.lower()
    return ' '.join(re.sub(u"[^a-zA-Z ]", "", text,  flags=re.UNICODE).split())

mp3_files_by_question_mapper = pickle.load(open('mp3_files_by_question.pkl', 'rb'))

textual_question = 'Is this a modern interior?'
mp3_files_by_question_mapper[clean_question(textual_question)]
# Output: 'complementary_0000010.mp3'

textual_question = 'Where can milk be obtained?'
mp3_files_by_question_mapper[clean_question(textual_question)]
# Output: 'complementary_0000011.mp3'

textual_question = 'What are the payment method of the parking meter?'
mp3_files_by_question_mapper[clean_question(textual_question)]
# Output: 'complementary_0000012.mp3'
```


### Image files
These links were taken from the [VQA Website](https://visualqa.org/download.html)
- train2014 images: [Download](http://images.cocodataset.org/zips/train2014.zip)
- val2014 images: [Download](http://images.cocodataset.org/zips/val2014.zip)
- test2015 images: [Download](http://images.cocodataset.org/zips/test2015.zip)


### Precomputed features
- BLIP features (train2014 images): [Download](https://drive.google.com/file/d/1-AR0Krjip2SYaKWY6dQvAhamVo91pUiJ/view?usp=sharing)
- BLIP features (val2014 images): [Download](https://drive.google.com/file/d/1-Q3dDlRue9dbDV3qwbGDaF6GNLm4rN9U/view?usp=sharing)
- Speech features of the whole SBVQA 2.0 dataset (Joanna only): [Download](https://drive.google.com/file/d/1Icdcw4rYyTzm4X3osAKNrsTXkyEHuMEq/view?usp=sharing)

## Pretrained Models
1. NeMo speech encoder checkpoint: [stt_en_conformer_ctc_large_24500_hours_bpe.nemo](https://drive.google.com/file/d/13mX_X3aS4MI_JX1UtYU0mIxXjaJFAzdZ/view)
2. Best SBVQA 2.0 checkpoint: [best_sbvqa_2.0_model.pt](https://drive.google.com/file/d/13j7T8dpKcjD499e6DGm4RL4UYjl1Kvvu/view)


## Authors

-   **Faris Alasmary** - [farisalasmary](https://github.com/farisalasmary)

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/farisalasmary/sbvqa2.0/blob/master/LICENSE) file for details

## Resources
- This code is mainly adapted from this repo: [Bottom-Up and Top-Down Attention for Visual Question Answering](https://github.com/hengyuan-hu/bottom-up-attention-vqa)
- NeMo Conformer checkpoint we used to develop the model: [Download](https://drive.google.com/file/d/1-FD-pLvCSy_TZ7POQap_XzpzsuvDnIBv/view?usp=sharing)
- BLIP model checkpoint finetuned on image captioning used in this repo: [Download](https://drive.google.com/file/d/1f0W9YWAC_N28WxLO2D-b27csYfHQkBmC/view?usp=sharing)
- VGG19 pretrained used in the SBVQA 1.0 implementation: [Download](https://drive.google.com/file/d/11S80FXLrVvpFQyHrwLBQiZ-cfQ5YqePs/view?usp=sharing)


### ToDo

- [x] speech feature extraction script (NeMo Conformer)
- [x] noise injection script
- [x] inference script
- [ ] visual feature extraction script (BLIP ViT)
- [x] main model training scripts
- [ ] upload find_the_best_speech_encoder.py script
- [ ] our SBVQA 1.0 implementation scripts
- [x] visualization scripts (GradCAM + attention maps)
- [x] upload SBVQA 2.0 dataset
- [x] upload precomputed visual and speech features
- [x] upload our pretrained models


## Citation

```
@article{alasmary2023sbvqa,
	author={Alasmary, Faris and Al-Ahmadi, Saad},
	journal={IEEE Access},
	title={SBVQA 2.0: Robust End-to-End Speech-Based Visual Question Answering for Open-Ended Questions},
	year={2023},
	volume={11},
	number={},
	pages={140967-140980},
	doi={10.1109/ACCESS.2023.3339537}
}
```
