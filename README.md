# First project - BirdCLEF classification

## Resources

### Audio Feature Extraction

- [Music Feature Extraction in Python](https://towardsdatascience.com/extract-features-of-music-75a3f9bc265d)
- [Audio Feature Extraction](https://devopedia.org/audio-feature-extraction#:~:text=Audio%20feature%20extraction%20is%20a,is%20a%20representation%20of%20sound.)
- TorchAudio

### GitHub Repos

- [pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis)
- [AudioClassification - PyTorch](https://github.com/yeyupiaoling/AudioClassification-Pytorch)
- [towhee](https://github.com/towhee-io/towhee)


## TODO

- [x] Add zero-crossing analysis to `data_exploration.Data.audio_features` - [source](https://www.analyticsvidhya.com/blog/2022/01/analysis-of-zero-crossing-rates-of-different-music-genre-tracks/)
- [x] Generate all the other metrics for all audios.
- [ ] Create different yet similar analyses to zero-crossing. For example, zero-max-mean line crossing, RMS line crossing.
- [ ] [Calculate zero-crossing rate](https://speechprocessingbook.aalto.fi/Representations/Zero-crossing_rate.html#:~:text=To%20calculate%20of%20the%20zero,length%20signal%20you%20need%20operations.)
- [ ] Train Models
  - [ ] Hierarchical Training: 
    - step1: 0-10, 10-20, ... models
    - step2: model1: 0:0-10, 1:10-20, 2:20-30, 3:30-40, 4:40-50, model2: ...
    - ...