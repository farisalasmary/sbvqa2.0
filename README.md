# SBVQA 2.0 Official Implementation
This is the official implementation of our paper:
> [SBVQA 2.0: Robust End-to-End Speech-Based Visual Question Answering for Open-Ended Questions](https://ieeexplore.ieee.org/document/10343139)


## How to run?
Coming soon!

## Data
### Audio files
Coming soon!

### Image files
Coming soon!

### Precomputed features
Coming soon!

## Pretrained Models
Coming soon!


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
- [ ] visual feature extraction script (BLIP ViT)
- [ ] main model training scripts
- [ ] find the best audio model script
- [ ] our SBVQA 1.0 implementation scripts
- [ ] visualization scripts (GradCAM + attention maps)
- [ ] upload SBVQA 2.0 dataset
- [ ] upload precomputed visual and speech features
- [ ] upload our pretrained models


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