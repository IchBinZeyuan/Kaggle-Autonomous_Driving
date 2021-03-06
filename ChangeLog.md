# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.8] -2019-12-28

### Added

- Add focal loss on mask

### Fixed

- Heatmap now works well

## [0.0.7] -2019-12-17

### Added

- Data Agumentation including blur, noise and flip
- Heatmap of mask
- Now we can decide if to use optimization of prediction (shorten time)
- Another base model: wide-resnet-101
- Use mask dataset information

### Changed

- Delete additional side of input image
- Increase threshold of sigmoid output
- Use input size of 512, batch size of 8

### Fixed
- Flip function works well now

## [0.0.6] -2019-11-28

### Added

- Stolen figure saving method from Kailiang.
- Normalize image in DataProcessing module.
- Assign weight on regr loss and mask loss respectively.

### Changed

- Save figures instead of showing them.
- Add regularization in model.

### Fixed

- Make resnet really learnable.

## [0.0.5] -2019-11-27

### Fixed

- Add system path. Now the project can be ran in terminal.


## [0.0.4] - 2019-11-26

### Changed

- Make upsample layer learnable.
- Add visulization function. 
- Fix some minor bugs.
- Improve structure of project.

### Results

- Submit ```resnetx101-Unet``` without learnable upsampling layer, get score of ```0.077```. :hankey:
- Submit ```resnetx101-Unet``` with learnable upsampling layer, get score of ```0.103```. :blush:

## [0.0.3] - 2019-11-24

### Changed 

- Change efficient-net to resnetx101.


## [0.0.2] - 2019-11-22

### Added

- Reconstruct [CenterNet_Baseline](https://www.kaggle.com/hocop1/centernet-baseline) to be the baseline of our project.

## [0.0.1] - 2019-11-20

### Added

- Run notebook stolen from [CenterNet_Baseline](https://www.kaggle.com/hocop1/centernet-baseline).