# Tweet-Disaster-Classification-with-BERT-and-Logistic-Regression
Tweet Disaster Classification with BERT and Logistic Regression.

## Introduction
This project focuses on the development of a tweet disaster classification system that leverages BERT (Bidirectional Encoder Representations from Transformers) and Logistic Regression to classify tweets into disaster-related and non-disaster-related categories. Initially, a baseline model was created using Logistic Regression to understand the basic trands and features in the data. The dataset is sourced from a Kaggle competition with as initiative aims to harness the potential of Natural Language Processing in emergency management and response. You can access the competition through the following link:  [Kaggle competition Overview](https://www.kaggle.com/competitions/nlp-getting-started/overview).


## Dataset

The dataset consists of tweets labeled as disaster-related or not. The objective is to develop a model that can accurately classify these tweets, which can be an invaluable tool for disaster relief organizations and news agencies.

- `train.csv`: Contains the training data with labels.
- `test.csv`: Contains the testing data without labels.

## Prerequisites
- Python 3.x
- Pandas
- Transformers
- Matplotlib
- Scikit-learn
- TensorFlow (for BERT)
- accelerate

To install all the necessary packages, you can use the following commands:
```shell
pip install transformers
pip install accelerate
pip install torch
pip install scikit-learn
pip install pandas
```

## Structure
The script consists of the following sections:

1. Data Preprocessing: The raw data undergoes several preprocessing steps including tokenization and encoding to prepare it for training.
2. Model Customization: A custom BERT classification model is created by modifying the classifier layer of the pre-trained BERT model.
3. Training and Validation: The custom model is trained on a subset of the data, and its performance is validated on a separate subset.
4. Testing and Submission: The trained model makes predictions on a test dataset, and the predictions are saved to a CSV file for submission.

## Usage
### Training the Model
1. Split the data into training and validation sets using a 90-10 split.
2. Preprocess and tokenize the data using the tokenize_function.
3. Create datasets for training, validation, and testing using the TweetDataset class.
4. Define training arguments using the TrainingArguments class.
5. Train the model using the Trainer class with the specified training arguments and datasets.
6. Save the best model obtained during training.

### Testing and Submission
1. Load the best model saved during training.
2. Make predictions on the test dataset using a new Trainer instance.
3. Save the predictions to a CSV file for submission.

## Results

The model's performance can be assessed using the accuracy metric obtained during the validation phase. The final predictions are saved to a submission.csv file, which can be used for submission in a competition or for further analysis.

During the evaluation phase, it was observed that both the Logistic Regression model, which serves as a baseline model, and the more complex BERT model produced similar results in terms of F1-score. This showcases the robustness of the Logistic Regression model for this specific task and poses interesting observations regarding the complexity-performance trade-off.

Here are the detailed accuracy scores for each model:
- Logistic Regression Model: 0.769
- BERT Model: 0.806

## Aknowledgements
The BERT model and tokenizer are sourced from the transformers library developed by Hugging Face
