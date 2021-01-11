# Introduction
* This is a competition using doctor-patient dialogue finding privacy sensitive span.

# Model Structure
* Input doctor-patient dialogue (split by speaker) through language model.
* Each word embedding go through 2 different feed-dorward network get BIO type and privacy type.
    * BIO type use for checking what span need to be output.
    * Privacy type use for checking what type of privacy sensitive to be output.

![](https://i.imgur.com/1HRzsQO.png)

# Usage
* All file have argument parser, please use `python3 FILENAME -h` to get more.

# Acknowledgments
* Thank [@eric88525](https://github.com/eric88525) doing data preprocessing (`preprocess_to_json.py`); [@HongYun0901](https://github.com/HongYun0901) pretraining the lenguage model.