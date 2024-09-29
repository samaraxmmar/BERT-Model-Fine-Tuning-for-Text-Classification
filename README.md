# BERT Model Fine-Tuning for Text Classification

**Fine-tuning BERT on User Queries for Action Classification**

---

## Table of Contents

1. [Project Description](#project-description)
2. [Dataset](#dataset)
3. [Features](#features)
4. [Technologies Used](#technologies-used)
5. [Installation](#installation)
6. [Training Process](#training-process)
7. [Usage](#usage)
8. [Results](#results)
9. [Contact](#contact)

---

## Project Description

This project involves fine-tuning the BERT (Bidirectional Encoder Representations from Transformers) model on a custom dataset designed for text classification tasks. The objective is to classify user queries into predefined actions, such as checking balance or making transfers, thereby enabling an automated response system for banking or finance applications.

## Dataset

The dataset consists of user queries paired with their respective actions. Each entry contains:

- **Text**: The user's query (e.g., "Peux-tu m'indiquer le solde actuel de mon compte").
- **Action**: The corresponding action category (e.g., "consulter solde").

### Sample Data

| Text Action | User Query |
| --- | --- |
| consulter solde | Peux-tu m'indiquer le solde actuel de mon compte |
| consulter solde | Donne-moi une mise à jour de mon compte |
| consulter solde | Affiche le montant actuel de mon compte |
| virement ccp | Envoyer des fonds depuis mon CCP pour des frais |
| virement ccp | Faire un transfert pour des frais de mise à jour |

**Note**: The dataset is structured to facilitate the classification of various user intents, which is crucial for developing an effective conversational agent.

## Features

- **Text Preprocessing**: Implementations for cleaning and tokenizing the text.
- **BERT Fine-Tuning**: Adjusts a pre-trained BERT model on the custom dataset for text classification.
- **Evaluation Metrics**: Calculates accuracy, precision, recall, and F1 score to assess model performance.
- **Model Persistence**: Saves the trained model for future predictions.

## Technologies Used

- **Python**: Programming language used for implementation.
- **PyTorch**: Framework utilized for deep learning.
- **Transformers**: Hugging Face's library for NLP models, particularly BERT.
- **CUDA**: For accelerated training on compatible GPUs.
- **Pandas and NumPy**: For data manipulation and preprocessing.

## Installation

To set up the project locally:

1. Clone the repository:
    ```sh
    git clone https://github.com/your-username/bert-finetuning-text-classification.git
    ```
2. Navigate to the project directory:
    ```sh
    cd bert-finetuning-text-classification
    ```
3. Install the required dependencies:
    ```sh
    python -m venv .venv
    source .venv/bin/activate  # For Linux/macOS
    .venv\Scripts\activate     # For Windows
    pip install -r requirements.txt
    ```

## Training Process

To fine-tune the BERT model:

1. **Prepare the Dataset**:
   - Place the dataset file (CSV) in the designated directory.
   - Ensure the script points to the correct dataset path.

2. **Run the Training Script**:
    ```sh
    python train.py
    ```
   This command will initiate the fine-tuning process and save the best-performing model based on validation performance.

## Usage

To utilize the trained model for predictions:

1. **Load the Fine-Tuned Model**:
    ```python
    from transformers import BertForSequenceClassification, BertTokenizer

    model = BertForSequenceClassification.from_pretrained('path/to/saved/model')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    ```

2. **Run Predictions**:
    ```python
    user_query = "Peux-tu vérifier combien il me reste actuellement?"
    inputs = tokenizer(user_query, return_tensors='pt')
    outputs = model(**inputs)
    predicted_action = outputs.logits.argmax(dim=-1)
    ```

## Results

After training and evaluating the model, the following metrics were achieved:

- **Accuracy**: XX%
- **Precision**: XX%
- **Recall**: XX%
- **F1 Score**: XX%

These results demonstrate the effectiveness of the fine-tuned model in classifying user intents accurately.

## Contact

For questions or collaboration opportunities:

- **Name**: Samar Ammar
- **Email**: (samarammar070@gmail.com)
