# Sexism Identification in Memes

This project focuses on automatically identifying sexist content in memes using machine learning approaches, specifically fine-tuned transformer models. The work is based on the EXIST 2024 dataset and implements multiple approaches for sexism detection in both English and Spanish memes.

## Project Overview

The project addresses the challenge of detecting sexism in multimodal content (memes) by combining image descriptions generated using GPT-4 with text content, then training transformer models for binary classification.

### Key Features

- **Multimodal Approach**: Combines visual content (through GPT-4 generated descriptions) with textual content
- **Multilingual Support**: Handles both English and Spanish memes
- **Multiple Model Architectures**: Implements BERT-ES, mBERT, and XLM-RoBERTa models
- **Comprehensive Evaluation**: Includes detailed performance metrics and analysis

## Dataset

The project uses the **EXIST 2024 Memes Dataset** which contains:
- Training set: ~4,020 memes (Spanish: ~2,010, English: ~2,010)
- Test set: ~880 memes (Spanish: ~540, English: ~340)
- Binary labels: YES (sexist) / NO (not sexist)

### Data Processing Pipeline

1. **Meme Separation**: Automatic separation of Spanish and English memes based on ID prefixes
2. **Description Generation**: GPT-4 Vision API generates textual descriptions of meme images
3. **Data Augmentation**: Combines original text with generated descriptions
4. **Label Processing**: Maps categorical labels to binary format

## Models and Approaches

### 1. Data Processing 

- Processes and extracts the EXIST 2024 dataset
- Separates memes by language (Spanish/English)
- Uses GPT-4 Vision API to generate meme descriptions
- Creates structured datasets with combined text and visual information

### 2. BERT-ES Fine-tuning 
- Implements fine-tuning of Spanish BERT model (`dccuchile/bert-base-spanish-wwm-cased`)
- Uses Optuna for hyperparameter optimization
- Includes comprehensive evaluation metrics

### 3. Multilingual Models

- Fine-tunes multilingual BERT (mBERT) and XLM-RoBERTa
- Supports both language-specific and combined training

**Models Implemented:**
- **mBERT** (`bert-base-multilingual-cased`)
- **XLM-RoBERTa** (`xlm-roberta-base`)


### 4. Evaluation 

## Installation and Setup

### Prerequisites

```bash
pip install transformers torch datasets
pip install sentencepiece accelerate optuna
pip install openai==0.28
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Project Structure

```
Sexism_Identification/
├── 01_data_processing.ipynb          # Data processing and preprocessing
├── 02_fine-tuning_bert-es.ipynb      # Spanish BERT fine-tuning
├── 02_fine-tuning_mBERT_XLM-RoBERTa.ipynb  # Multilingual models
├── 03_evaluate.ipynb                 # Model evaluation and analysis
├── data/                             # Dataset files and processed data
├── results/                          # Model outputs and predictions
└── README.md                         # Project documentation
```

## Technical Details

### Model Architecture

- **Input**: Combined text (original + GPT-4 description)
- **Tokenization**: Language-specific tokenizers
- **Classification Head**: Binary classification layer
- **Training**: Fine-tuning with early stopping

### Data Augmentation Strategy

1. **Visual Information**: GPT-4 generated image descriptions
2. **Text Combination**: Concatenation of original text + descriptions
3. **Language-Specific Processing**: Tailored prompts for Spanish/English

## License

This project is for research and educational purposes. Please refer to the EXIST 2024 dataset license for data usage terms.

## Acknowledgments

- EXIST 2024 shared task organizers
- Hugging Face transformers library
- OpenAI GPT-4 Vision API
- University of Zurich research support

## Contact

For questions or collaboration opportunities, please refer to the project repository or contact the research team.
