<p align="center">
  <img src="[https://static.vecteezy.com/ti/vecteur-libre/t2/21495252-banniere-de-technologie-numerique-concept-d-arriere-plan-bleu-vert-effet-de-lumiere-de-cyber-technologie-technologie-abstraite-donnees-futures-d-innovation-reseau-internet-donnees-ai-big-connexion-de-points-de-lignes-vecteur-d-illustration-vectoriel.jpg](https://img.freepik.com/vecteurs-premium/fond-abstrait-geometrique-connecte-points-lignes-connexion-contexte-technologique_322958-3166.jpg)" alt="Project Banner" width="800"/>
</p>

<h1 align="center">Fine-tuning BERT for Action Classification</h1>

<p align="center">
  <a href="https://www.python.org/downloads/release/python-390/"><img src="https://img.shields.io/badge/Python-3.9%2B-blue.svg" alt="Python Version"></a>
  <a href="https://pytorch.org/get-started/locally/"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg" alt="PyTorch Version"></a>
  <a href="https://huggingface.co/docs/transformers/index"><img src="https://img.shields.io/badge/Transformers-4.3%2B-yellow.svg" alt="Transformers Version"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
</p>

---

## 🎯 Project Description

This project demonstrates the fine-tuning of a pre-trained **BERT (Bidirectional Encoder Representations from Transformers)** model for a text classification task. The primary goal is to classify user queries from a banking or financial context into predefined actions, such as checking an account balance or initiating a transfer. This enables the development of a sophisticated and automated response system, enhancing user interaction with financial applications.

---

## 🗂️ Dataset

The model is trained on a custom dataset of user queries paired with their corresponding actions. Each entry in the dataset includes the user's query text and the action it represents.

### Sample Data

| Text Action | User Query |
| :--- | :--- |
| `consulter solde` | "Peux-tu m'indiquer le solde actuel de mon compte" |
| `consulter solde` | "Donne-moi une mise à jour de mon compte" |
| `consulter solde` | "Affiche le montant actuel de mon compte" |
| `virement ccp` | "Envoyer des fonds depuis mon CCP pour des frais" |
| `virement ccp` | "Faire un transfert pour des frais de mise à jour" |

This structured dataset is key to training a model that can accurately understand and categorize a wide range of user intents, which is a critical component for building an effective conversational agent.

---

## ✨ Features

-   **Text Preprocessing**: Includes scripts for cleaning, normalizing, and tokenizing text data to prepare it for the BERT model.
-   **BERT Fine-Tuning**: Leverages a pre-trained multilingual BERT model and fine-tunes it on the specific task of action classification.
-   **Performance Evaluation**: Implements standard evaluation metrics such as Accuracy, Precision, Recall, and F1-score to rigorously assess the model's performance.
-   **Model Persistence**: The trained model is saved, allowing for easy integration into other applications for real-time inference.

---

## 🚀 Technologies Used

<p align="center">
  <img src="https://i.imgur.com/t4J2T1h.png" alt="Technologies" width="600"/>
</p>

-   **Python**: The core programming language for the project.
-   **PyTorch**: A powerful deep learning framework used for building and training the model.
-   **Hugging Face Transformers**: Provides the pre-trained BERT model and the necessary tools for fine-tuning.
-   **CUDA**: Utilized for GPU acceleration to speed up the training process.
-   **Pandas & NumPy**: Essential libraries for data manipulation and numerical operations.

---

## 🛠️ Installation

To get the project up and running on your local machine, please follow these steps:

1.  **Clone the Repository**
    ```sh
    git clone [https://github.com/samaraxmmar/BERT-Model-Fine-Tuning-for-Text-Classification.git](https://github.com/samaraxmmar/BERT-Model-Fine-Tuning-for-Text-Classification.git)
    ```
2.  **Navigate to the Project Directory**
    ```sh
    cd BERT-Model-Fine-Tuning-for-Text-Classification
    ```
3.  **Set Up a Virtual Environment and Install Dependencies**
    ```sh
    python -m venv .venv
    # On Linux/macOS
    source .venv/bin/activate
    # On Windows
    .venv\Scripts\activate
    pip install -r requirements.txt
    ```

---

## 🧠 Training Process

The fine-tuning process is managed through the `bert.ipynb` Jupyter Notebook.

<p align="center">
  <img src="https://i.imgur.com/3iGj2yK.gif" alt="Training Process" width="600"/>
</p>

1.  **Prepare Your Dataset**:
    -   Place your training data (in CSV format) into the project's data directory.
    -   Update the file path in the notebook to point to your dataset.

2.  **Execute the Notebook**:
    -   Open `bert.ipynb` in Jupyter Notebook or Google Colab.
    -   Run the cells sequentially to load the data, preprocess it, and initiate the training loop. The following snippet loads the pre-trained BERT model for sequence classification:
        ```python
        from transformers import BertForSequenceClassification

        # Load the model with the specified number of labels
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-multilingual-uncased",
            num_labels=NUM_LABELS,
            id2label=id2label,
            label2id=label2id
        )
        model.to(device)
        ```
    The notebook will handle the training and save the model with the best performance on the validation set.

---

## ⚙️ Usage

Once the model is trained and saved, you can easily use it for inference on new user queries.

1.  **Load the Fine-Tuned Model and Tokenizer**:
    ```python
    from transformers import BertForSequenceClassification, BertTokenizer

    model_path = 'path/to/your/saved/model'
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    ```

2.  **Make Predictions**:
    ```python
    user_query = "Peux-tu vérifier combien il me reste actuellement?"
    inputs = tokenizer(user_query, return_tensors='pt', padding=True, truncation=True)

    # Get model outputs
    outputs = model(**inputs)

    # Get the predicted class
    predicted_action_id = outputs.logits.argmax(dim=-1).item()
    predicted_action = model.config.id2label[predicted_action_id]

    print(f"Predicted Action: {predicted_action}")
    ```

---

## 📊 Results

The fine-tuned model achieved exceptional performance on the test set, demonstrating its effectiveness in classifying user intents with high accuracy.

<p align="center">
  <img src="https://i.imgur.com/O6Yv8gQ.png" alt="Confusion Matrix" width="500"/>
</p>

### **Classification Report**

| Class | Precision | Recall | F1-Score | Support |
| :---: | :---: | :---: | :---: | :---: |
|   0   | 1.00  | 1.00  | 1.00  |  104  |
|   1   | 0.99  | 1.00  | 1.00  |  107  |
|   2   | 1.00  | 1.00  | 1.00  |   89  |
|   3   | 1.00  | 0.99  | 0.99  |   97  |

### **Overall Metrics**

-   **Accuracy**: **100%**
-   **Macro Average F1-Score**: **1.00**
-   **Weighted Average F1-Score**: **1.00**

These outstanding results confirm that the fine-tuned BERT model can serve as a highly reliable component in a customer service or financial chatbot application.

---

## 📫 Contact

For any questions, feedback, or collaboration opportunities, please feel free to reach out:

**Samar Ammar**

-   **Email**: `samarammar070@gmail.com`
-   **GitHub**: [samaraxmmar](https://github.com/samaraxmmar)
