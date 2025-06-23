# Cyberbullying Classification with BERT

## Overview

This repository gives a Jupyter notebook–based pipeline that fine-tunes a pre-trained BERT model (from Hugging Face) to classify tweets into multiple cyberbullying categories. It walks you through data loading, cleaning, oversampling, tokenization, model definition, training, evaluation, and saving the best model for downstream inference.

## Features

- **Data preprocessing**  
  - Cleans text (removes emojis, URLs, punctuation)  
  - Token-level normalization using spaCy and regex  
- **Handling class imbalance**  
  - Uses [imbalanced-learn](https://github.com/scikit-learn-contrib/imbalanced-learn) to oversample minority classes  
- **BERT fine-tuning**  
  - Custom PyTorch `Bert_Classifier` wrapping Hugging Face’s `bert-base-uncased`  
  - Configurable training hyperparameters (batch size, learning rate, epochs)  
- **Evaluation & Visualization**  
  - Reports accuracy, precision, recall, F1  
  - Displays confusion matrix and example predictions  
- **Model persistence**  
  - Saves fine-tuned model weights under `model_save/` for later inference  

## Getting Started

### Prerequisites

- **Python** 3.7 or higher  
- **Jupyter Notebook**  
- **CUDA-enabled GPU** (optional, for faster training)  

### Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/AnikaGBasu/CyCopDiscordBot.git
   cd CyCopDiscordBot
   ```

2. **Create & activate a virtual environment**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install required packages**  
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   If you don’t have a `requirements.txt`, you can install manually:
   ```bash
   pip install pandas numpy matplotlib seaborn nltk spacy emoji scikit-learn imbalanced-learn torch transformers
   python -m spacy download en_core_web_sm
   ```

4. **Download the dataset**  
   Place `cycop_bert_model_cyberbullying_tweets.csv` in the project root (same folder as `bertbot.ipynb`).

## Usage

1. **Launch Jupyter**  
   ```bash
   jupyter notebook bertbot.ipynb
   ```
2. **Follow the notebook cells** in order:
   1. Environment setup & imports  
   2. Data inspection & cleaning  
   3. Train-validation-test split & oversampling  
   4. Tokenization with `BertTokenizer`  
   5. Building `Bert_Classifier` class  
   6. Training loop (`bert_train`) with evaluation  
   7. Plotting metrics & confusion matrix  
   8. Saving the fine-tuned model to `model_save/`

3. **Run inference** on new tweets by loading the saved model and using the `bert_predict` function.

## Customization

- Adjust **hyperparameters** in the “Initialize Model” section (learning rate, epochs, batch size).
- Freeze BERT layers by setting `freeze_bert=True` when instantiating the classifier.
- Swap in a different Hugging Face model (e.g., `roberta-base`) by changing the tokenizer and base model calls.

## Contributing

1. Fork the repository  
2. Create a feature branch: `git checkout -b feature/YourFeature`  
3. Commit your changes: `git commit -m "Add <feature>"`  
4. Push and open a Pull Request  

Please ensure that all new code is covered by tests and adheres to PEP8 styling.

## Contact

Feel free to contact anika.ghosh.basu@gmail.com with any comments. 
