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

### Dataset Setup

Please manually download the required datasets and place them in the appropriate directories:
- **EXIST 2024 Memes Dataset**: Download and extract to `./data/` directory
- **Pre-processed training data**: Ensure processed files are in `./data/` directory  
- **GPT-4 generated descriptions**: Generated files will be saved to `./data/` directory

**Required Directory Structure:**
```
data/
├── EXIST 2024 Memes Dataset/
├── Spanish_memes/
├── English_memes/
├── Spanish_memes_test/
├── English_memes_test/
└── [processed JSON files]

results/
└── [model outputs and predictions]
```

### Data Processing Pipeline

1. **Meme Separation**: Automatic separation of Spanish and English memes based on ID prefixes
2. **Description Generation**: GPT-4 Vision API generates textual descriptions of meme images
3. **Data Augmentation**: Combines original text with generated descriptions
4. **Label Processing**: Maps categorical labels to binary format

## Models and Approaches

### 1. Data Processing (`01_data_processing.ipynb`)

- Processes and extracts the EXIST 2024 dataset
- Separates memes by language (Spanish/English)
- Uses GPT-4 Vision API to generate meme descriptions
- Creates structured datasets with combined text and visual information

**Key Functions:**
- `analyze_image_for_sexist_content()`: GPT-4 based image analysis
- `create_df()`: Batch processing of memes with error handling
- Language-specific prompt engineering for Spanish and English

**Note**: Requires OpenAI API key for GPT-4 Vision access.

### 2. BERT-ES Fine-tuning (`02_fine-tuning_bert-es.ipynb`)

- Implements fine-tuning of Spanish BERT model (`dccuchile/bert-base-spanish-wwm-cased`)
- Uses Optuna for hyperparameter optimization
- Includes comprehensive evaluation metrics

**Key Features:**
- Hyperparameter optimization with 15 trials
- Early stopping to prevent overfitting
- Detailed performance metrics (accuracy, F1, precision, recall)
- Confusion matrix analysis

### 3. Multilingual Models (`02_fine-tuning_mBERT_XLM-RoBERTa.ipynb`)

- Fine-tunes multilingual BERT (mBERT) and XLM-RoBERTa
- Supports both language-specific and combined training
- Comparative analysis across different model architectures

**Models Implemented:**
- **mBERT** (`bert-base-multilingual-cased`)
- **XLM-RoBERTa** (`xlm-roberta-base`)


### 4. Evaluation (`03_evaluate.ipynb`)

- Comprehensive performance evaluation
- Confusion matrix visualization
- Detailed classification reports
- Cross-model performance comparison

## Installation and Setup

### Prerequisites

```bash
pip install transformers torch datasets
pip install sentencepiece accelerate optuna
pip install openai==0.28
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Usage

### 1. Data Processing

Run `01_data_processing.ipynb` to:
- Process and extract the EXIST 2024 dataset
- Generate meme descriptions using GPT-4
- Create processed datasets for training

**Note**: Requires OpenAI API key for GPT-4 Vision access.

### 2. Model Training

#### Spanish BERT:
```python
# Run 02_fine-tuning_bert-es.ipynb
# Automatically optimizes hyperparameters and trains model
```

#### Multilingual Models:
```python
# Run 02_fine-tuning_mBERT_XLM-RoBERTa.ipynb
# Trains both mBERT and XLM-RoBERTa on Spanish and English data
```

### 3. Evaluation

Run `03_evaluate.ipynb` for comprehensive model evaluation and visualization.

## Results

### Performance Overview

The models achieve competitive performance on the sexism detection task:

- **BERT-ES**: Optimized specifically for Spanish memes
- **mBERT**: Cross-lingual performance with multilingual training
- **XLM-RoBERTa**: State-of-the-art multilingual transformer performance

### Key Metrics

- **Accuracy**: ~72-75% on test set
- **F1-Score**: ~78% for positive class (sexist content)
- **Precision/Recall**: Balanced performance across classes

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

### Hyperparameter Optimization

Uses Optuna for automated hyperparameter tuning:
- Learning rate: 1e-5 to 5e-5
- Batch size: 8, 16
- Epochs: 3-5
- Dropout rate: 0-0.5
- Weight decay: 0-0.3

### Data Augmentation Strategy

1. **Visual Information**: GPT-4 generated image descriptions
2. **Text Combination**: Concatenation of original text + descriptions
3. **Language-Specific Processing**: Tailored prompts for Spanish/English

## Future Work

- **Multimodal Integration**: Direct image feature extraction
- **Advanced Architectures**: Vision-Language models (CLIP, BLIP)
- **Data Expansion**: Additional languages and cultural contexts
- **Real-time Detection**: Deployment-ready inference pipeline

## Contributing

This project is part of academic research on automated content moderation and bias detection. Contributions are welcome for:
- Model improvements
- Additional language support
- Evaluation metrics
- Documentation enhancements

## License

This project is for research and educational purposes. Please refer to the EXIST 2024 dataset license for data usage terms.

## Acknowledgments

- EXIST 2024 shared task organizers
- Hugging Face transformers library
- OpenAI GPT-4 Vision API
- University of Zurich research support

## Contact

For questions or collaboration opportunities, please refer to the project repository or contact the research team.