# T2C_MoviePosters

TODO: Change the palette output format

### Dataset

`TPN_dataset/metadata` : Contains the movie details in json formats. Use the notebook to convert into following dataset.

`TPN_dataset/traindata` : Contains all, train and test data for genres, movie names, plots and palettes.

### Samples

colorThief palette examples at https://drive.google.com/open?id=1DeUluYLUbki8lRsbj2nu9zD1imz2JTKO.

`TPN_samples/safe_RGB_testset` : testing set output on model trained (2000 epochs) on safe RGB palettes (using colorThief). More at https://drive.google.com/open?id=1ZLfzIXWAZ_RFz9g7UWZgFbtwFfnzYHNt.

`TPN_samples/safe_RGB_genres` : individual genres names output on model trained (2000 epochs) on safe RGB palettes (using colorThief). Also at https://drive.google.com/drive/folders/1QQI4QwEnGkxUe5YfYKryyxvMc1D_FYHa?usp=sharing.

270 Epochs trained on only the plots with minimal preprocessing at https://imgur.com/a/T6YYvTq

### Models

`TPN_models/safe_RGB` : 2000 epoch trained models for TPN on safe RGB palettes

### Source

`SRC/Palette_Generator` : Code for generating the palettes using https://gfx.cs.princeton.edu/pubs/Chang_2015_PPR/chang2015-palette_small.pdf - Direct port of available code in JS
