# text-lang-recognition-model
DNN text language recognition model

## General
This is a utility project for <a href='https://github.com/DragunovK/language-recognition'>Language Recognition project</a>.
It can be used to prepare a model for DNN method of text language recognition as well as to prepare language profiles for n-gram method of text language recognition.
It contains 2 utility scripts: 
* prepare_model.py -- For teaching neural network model
* prepare_ngram.py -- For preparing language profiles for n-gram method

## Technologies
* NLTK
* scikit-learn
* Tensorflow
* NumPy
* Pandas

## How to use
* Clone this repository
  ```
  git clone https://github.com/DragunovK/text-lang-recognition-model.git
  ```
* Install dependencies
  ```
  pip install requirements.txt
  ```
* Depends on which script you're using:
  * prepare_model.py: 
    * You can edit langs list, output_dir variable, model_name variable. 
      Default values: 
      ```
      langs=['eng', 'rus']
      output_dir = 'output/neural'
      model_name = 'en_ru_model'
      ```
    * Run the script
      ```
      python .\prepare_model.py
      ```
  * prepare_ngram.py:
    * You can edit You can edit langs list, output_dir variable.
      Default values: 
      ```
      langs=['eng', 'rus']
      output_dir = 'output/ngram'
      ```
    * Run the script
      ```
      python .\prepare_ngram.py
      ```
