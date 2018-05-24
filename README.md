## QANet in keras
QANet: https://arxiv.org/abs/1804.09541

This keras model refers to QANet in tensorflow (https://github.com/NLPLearn/QANet), and the self-attention & position embedding are used from (https://kexue.fm/archives/4765, https://github.com/bojone/attention).

## Pipline
1. Download squad data from (https://rajpurkar.github.io/SQuAD-explorer/).

2. Run `preprocess.ipynb` and `handcraft.ipynb` to get npys of the preprocessed data and handcraft features.

3. Run `train_QANet.py` to start training.

4. Fast demo: Use the god made `model.fit()` in `QANet_fit_demo.py` with random numpy data.

## Updates
- [x] Add EMA
- [x] Add multi gpu (speed up)
- [x] Support adding handcraft features
- [ ] Add layer dropout
- [ ] Add data augmentation

## Results
Todo...
