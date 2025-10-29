# Georgian Attractions Enrichment Pipeline

 **Automated enrichment pipeline for tourist attractions dataset**

[![HuggingFace](https://img.shields.io/badge/🤗-Dataset-yellow)](https://huggingface.co/datasets/AIAnastasia/Georgian-attractions)

##  About This Project

This is a project demonstrating how to build an NLP enrichment pipeline for processing tourist attraction descriptions. The pipeline was originally used to create the [Georgian Attractions dataset](https://huggingface.co/datasets/AIAnastasia/Georgian-attractions) on HuggingFace.

###  Purpose

This project showcases:
-  Named Entity Recognition (NER) for location/organization extraction
-  Zero-shot classification for categorization
-  Tag generation using transformer models
-  Fuzzy string matching for multilingual data alignment
-  Professional Python project structure
-  Modular, testable, documented code

###  Final Dataset

The enriched dataset is **already available** on HuggingFace:
-  **Dataset**: [AIAnastasia/Georgian-attractions](https://huggingface.co/datasets/AIAnastasia/Georgian-attractions)
-  **Size**: 1,628 attractions (Russian & English)
-  **Features**: name, description, category, NER entities, tags, location, images
-  **Status**: Production-ready, actively used

##  Use Cases

### For This Pipeline

1. **Process NEW attraction data** (not yet enriched)
2. **Learn NLP pipeline architecture**
3. **Adapt for similar datasets** (hotels, restaurants, etc.)
4. **Understand multilingual NLP processing**

### For the Dataset

The enriched dataset on HuggingFace can be used for:
-  Building RAG (Retrieval-Augmented Generation) systems
-  Semantic search applications
-  Tourism chatbots
-  Travel recommendation apps

##  Architecture

```
Input Data                Pipeline Components              Output Data
────────────              ──────────────────              ───────────

name_ru         ──────►   NERProcessor        ──────►    + ner_ru
name_en         ──────►   CategoryClassifier  ──────►    + category_ru
description_ru  ──────►   TagGenerator        ──────►    + tags_ru
description_en  ──────►   FuzzyMatcher        ──────►    + category_en
                                                          + tags_en
                                                          + name_matches
```

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/yourusername/georgian-attractions-enrichment.git
cd georgian-attractions-enrichment

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

##  Quick Start

### Option 1: Process Sample Data (2 minutes)

```bash
# Test the pipeline on sample data
python test_quick.py
```

### Option 2: Process New Data

```python
from src.pipelines.enrichment_pipeline import EnrichmentPipeline
import pandas as pd

# Prepare your data in classic format
new_data = pd.DataFrame({
    'name_ru': ['Новая крепость'],
    'name_en': ['New Fortress'],
    'description_ru': ['Древняя крепость на горе...'],
    'description_en': ['Ancient fortress on the mountain...']
})

# Save and process
new_data.to_csv('./data/raw/new_attractions.csv', index=False)

# Run enrichment
pipeline = EnrichmentPipeline('config/config.yaml')
enriched = pipeline.run(source='./data/raw/new_attractions.csv')
```

### Option 3: Use the Existing Dataset

```python
from datasets import load_dataset

# Load the already-enriched dataset from HuggingFace
dataset = load_dataset("AIAnastasia/Georgian-attractions")
df = dataset['train'].to_pandas()

print(f"Loaded {len(df)} enriched attractions")
print(df.columns.tolist())
# ['id', 'name', 'description', 'location', 'category', 'ner', 'language', 'image']
```

##  Components

### 1. NER Processor
Extracts named entities (locations, organizations, persons):

```python
from src.enrichment.ner_processor import NERProcessor

processor = NERProcessor(config)
entities = processor.extract_entities(
    "Narikala Fortress is located in Tbilisi",
    language='en'
)
# {'locations': ['Narikala', 'Tbilisi'], 'organizations': [], ...}
```

### 2. Category Classifier
Classifies into: nature, history, culture, spirituality, architecture:

```python
from src.enrichment.category_classifier import CategoryClassifier

classifier = CategoryClassifier(config)
category = classifier.classify(
    "Ancient fortress with mountain views",
    language='en'
)
# 'history'
```

### 3. Tag Generator
Generates relevant tags:

```python
from src.enrichment.tag_generator import TagGenerator

generator = TagGenerator(config)
tags = generator.generate_tags(
    "Medieval church in the mountains",
    language='en'
)
# ['church', 'medieval', 'mountain', 'historical', 'architecture']
```

### 4.FuzzyMatcher

```python
from src.utils.fuzzy_matching import FuzzyMatcher

matcher = FuzzyMatcher(config)
score = matcher.match("Нарикала", "Narikala")
```

##  Project Structure

```
georgian-attractions-enrichment/
├── config/
│   └── config.yaml              # Configuration
├── data/
│   ├── raw/                     # Input data
│   ├── processed/               # Enriched output
│   └── cache/                   # Model cache
├── src/
│   ├── enrichment/              # Core enrichment components
│   │   ├── ner_processor.py
│   │   ├── category_classifier.py
│   │   └── tag_generator.py
│   ├── utils/                   # Utilities
│   │   ├── data_loader.py
│   │   ├── fuzzy_matching.py
│   │   └── data_adapter.py      # Format converter
│   └── pipelines/
│       └── enrichment_pipeline.py  # Main pipeline
├── scripts/
│   └── run_enrichment.py        # CLI tool
├── tests/                       # Unit tests
├── main.py                      # Simple entry point
└── README.md
```


##  License

**Code**: MIT License (free to use, modify, distribute)

**Dataset**: The [Georgian Attractions dataset](https://huggingface.co/datasets/AIAnastasia/Georgian-attractions) is licensed for **educational purposes only**.

##  Acknowledgments

**Models:**
- NER: `dslim/bert-base-NER`, Davlan/xlm-roberta-base-ner-hrl
- Classification: `joeddav/xlm-roberta-large-xnli`

**Dataset:**
- Original data sourced from public tourism information
- Enriched using this pipeline
- Available on [HuggingFace](https://huggingface.co/datasets/AIAnastasia/Georgian-attractions)
