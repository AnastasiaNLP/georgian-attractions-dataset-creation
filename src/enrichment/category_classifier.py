"""
Category Classifier
Module for zero-shot classification of Georgian attractions into categories.
Supports multilingual classification using XLM-RoBERTa model.

Categories:
- nature: Natural landmarks, parks, mountains, lakes
- history: Historical sites, fortresses, ancient places
- culture: Museums, galleries, cultural centers
- spirituality: Churches, monasteries, temples, cathedrals
- architecture: Architectural monuments, buildings

Uses: joeddav/xlm-roberta-large-xnli model for zero-shot classification
"""

import logging
from typing import Dict, List, Any, Optional, Union
from transformers import pipeline
import pandas as pd
from datasets import Dataset

logger = logging.getLogger(__name__)


class CategoryClassifier:
    """
    Zero-shot classifier for categorizing attractions.

    Attributes:
        config (dict): Model configuration
        pipeline: Loaded classification pipeline
        categories (dict): Category labels for each language
        stats (dict): Classification statistics
    """

    def __init__(self, config: dict):
        """
        Initialize category classifier.

        Args:
            config: Dictionary with model and category configuration
                Example:
                {
                    'zero_shot': {
                        'model': 'joeddav/xlm-roberta-large-xnli',
                        'device': 0,
                        'batch_size': 32,
                        'hypothesis_template': 'This example is about {}.'
                    },
                    'categories': {
                        'en': ['nature', 'history', 'culture', 'spirituality', 'architecture'],
                        'ru': ['природа', 'история', 'культура', 'духовность', 'архитектура']
                    }
                }
        """
        self.config = config
        self.pipeline = None
        self.categories = config.get('categories', {})
        self.stats = {
            'processed': 0,
            'errors': 0,
            'category_counts': {}
        }

        logger.info("Initializing category classifier")
        self._load_model()

    def _load_model(self):
        """Load zero-shot classification model."""
        try:
            logger.info("Loading zero-shot classification model...")

            zero_shot_config = self.config.get('zero_shot', {})

            self.pipeline = pipeline(
                "zero-shot-classification",
                model=zero_shot_config.get('model', 'joeddav/xlm-roberta-large-xnli'),
                device=zero_shot_config.get('device', -1)
            )

            # Store batch size and hypothesis template
            self.batch_size = zero_shot_config.get('batch_size', 32)
            self.hypothesis_template = zero_shot_config.get(
                'hypothesis_template',
                'This example is about {}.'
            )

            logger.info(" Zero-shot classification model loaded")

        except Exception as e:
            logger.error(f"Error loading classification model: {e}")
            raise

    def classify(
        self,
        text: str,
        language: str = 'en',
        top_k: int = 1
    ) -> Union[str, List[str]]:
        """
        Classify single text into category.

        Args:
            text: Input text to classify
            language: Language of text ('en' or 'ru')
            top_k: Number of top categories to return

        Returns:
            If top_k=1: Single category string
            If top_k>1: List of top categories

        Example:
            >>> classifier = CategoryClassifier(config)
            >>> category = classifier.classify(
            ...     "Ancient fortress with beautiful views",
            ...     language='en'
            ... )
            >>> print(category)
            'history'
        """
        if not text or not isinstance(text, str):
            logger.warning("Empty or invalid text provided")
            return "" if top_k == 1 else []

        if language not in self.categories:
            logger.warning(f"Language {language} not supported")
            return "" if top_k == 1 else []

        try:
            candidate_labels = self.categories[language]

            result = self.pipeline(
                text,
                candidate_labels=candidate_labels,
                hypothesis_template=self.hypothesis_template
            )

            self.stats['processed'] += 1

            # Update category counts
            top_category = result['labels'][0]
            self.stats['category_counts'][top_category] = \
                self.stats['category_counts'].get(top_category, 0) + 1

            if top_k == 1:
                return result['labels'][0]
            else:
                return result['labels'][:top_k]

        except Exception as e:
            logger.error(f"Error classifying text: {e}")
            self.stats['errors'] += 1
            return "" if top_k == 1 else []

    def classify_batch(
        self,
        texts: List[str],
        language: str = 'en',
        top_k: int = 1
    ) -> List[Union[str, List[str]]]:
        """
        Classify multiple texts in batch.

        Args:
            texts: List of texts to classify
            language: Language of texts
            top_k: Number of top categories per text

        Returns:
            List of categories (or lists of categories if top_k>1)

        Example:
            >>> texts = [
            ...     "Beautiful mountain lake",
            ...     "Ancient church from 12th century"
            ... ]
            >>> categories = classifier.classify_batch(texts, language='en')
            >>> print(categories)
            ['nature', 'spirituality']
        """
        logger.info(f"Classifying batch of {len(texts)} texts...")

        results = []
        for idx, text in enumerate(texts):
            if idx % 100 == 0 and idx > 0:
                logger.info(f"Processed {idx}/{len(texts)} texts")

            category = self.classify(text, language, top_k)
            results.append(category)

        logger.info(f"✓ Batch classification completed")
        return results

    def enrich_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str,
        language: str = 'en',
        output_column: str = 'category',
        top_k: int = 1
    ) -> pd.DataFrame:
        """
        Enrich dataframe with category classifications.

        Args:
            df: Input dataframe
            text_column: Name of column with text to classify
            language: Language of texts
            output_column: Name of output column for categories
            top_k: Number of top categories to return

        Returns:
            Dataframe with added category column

        Example:
            >>> df = pd.DataFrame({
            ...     'name': ['Narikala', 'Lake Ritsa'],
            ...     'description': [
            ...         'Ancient fortress in Tbilisi',
            ...         'Beautiful mountain lake'
            ...     ]
            ... })
            >>> enriched_df = classifier.enrich_dataframe(
            ...     df,
            ...     text_column='description',
            ...     language='en'
            ... )
            >>> print(enriched_df['category'].tolist())
            ['history', 'nature']
        """
        logger.info(f"Enriching dataframe with categories ({len(df)} records)...")

        if text_column not in df.columns:
            raise ValueError(f"Column {text_column} not found in dataframe")

        # Classify each text
        categories = self.classify_batch(
            texts=df[text_column].tolist(),
            language=language,
            top_k=top_k
        )

        # Add to dataframe
        df[output_column] = categories

        logger.info(f"✓ Dataframe enrichment completed")
        return df

    def classify_dataset(
        self,
        dataset: Union[Dataset, pd.DataFrame],
        column: str,
        language: str = 'en',
        top_k: int = 1
    ) -> List[Union[str, List[str]]]:
        """
        Classify HuggingFace Dataset or DataFrame column.

        This method is optimized for large datasets and uses batch processing.

        Args:
            dataset: HuggingFace Dataset or pandas DataFrame
            column: Name of text column to classify
            language: Language of texts
            top_k: Number of top categories per text

        Returns:
            List of categories for entire dataset

        Example:
            >>> from datasets import load_dataset
            >>> dataset = load_dataset('AIAnastasia/Georgian-attractions')
            >>> categories = classifier.classify_dataset(
            ...     dataset['train'],
            ...     column='description_en',
            ...     language='en'
            ... )
        """
        logger.info(f"Classifying dataset column '{column}'...")

        # Convert to list of texts
        if isinstance(dataset, Dataset):
            texts = dataset[column]
        elif isinstance(dataset, pd.DataFrame):
            texts = dataset[column].tolist()
        else:
            raise TypeError("Dataset must be HuggingFace Dataset or pandas DataFrame")

        # Classify in batches
        return self.classify_batch(texts, language, top_k)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get classification statistics.

        Returns:
            Statistics dictionary with category distribution:
            {
                'processed': 500,
                'errors': 5,
                'category_counts': {
                    'nature': 150,
                    'history': 180,
                    'culture': 70,
                    'spirituality': 95,
                    'architecture': 5
                },
                'success_rate': 0.99
            }
        """
        success_rate = (
            (self.stats['processed'] - self.stats['errors']) / self.stats['processed']
            if self.stats['processed'] > 0 else 0
        )

        return {
            **self.stats,
            'success_rate': round(success_rate, 3)
        }

    def get_category_distribution(self) -> Dict[str, float]:
        """
        Get percentage distribution of categories.

        Returns:
            Dictionary with category percentages:
            {
                'nature': 30.0,
                'history': 36.0,
                'culture': 14.0,
                'spirituality': 19.0,
                'architecture': 1.0
            }
        """
        total = sum(self.stats['category_counts'].values())
        if total == 0:
            return {}

        distribution = {
            category: round((count / total) * 100, 2)
            for category, count in self.stats['category_counts'].items()
        }

        return dict(sorted(distribution.items(), key=lambda x: x[1], reverse=True))


# Usage example
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Example configuration
    config = {
        'zero_shot': {
            'model': 'joeddav/xlm-roberta-large-xnli',
            'device': -1,  # CPU
            'batch_size': 32,
            'hypothesis_template': 'This example is about {}.'
        },
        'categories': {
            'en': ['nature', 'history', 'culture', 'spirituality', 'architecture'],
            'ru': ['природа', 'история', 'культура', 'духовность', 'архитектура']
        }
    }

    # Initialize classifier
    classifier = CategoryClassifier(config)

    # Test single classification - English
    text_en = "Ancient fortress with beautiful mountain views and historical significance"
    category_en = classifier.classify(text_en, language='en')
    print("\nEnglish classification:")
    print(f"Text: {text_en}")
    print(f"Category: {category_en}")

    # Test single classification - Russian
    text_ru = "Древняя крепость с красивым видом на горы"
    category_ru = classifier.classify(text_ru, language='ru')
    print("\nRussian classification:")
    print(f"Text: {text_ru}")
    print(f"Category: {category_ru}")

    # Test batch classification
    texts_en = [
        "Beautiful mountain lake surrounded by forests",
        "Medieval church from 11th century",
        "National art museum with modern exhibitions"
    ]
    categories = classifier.classify_batch(texts_en, language='en')
    print("\nBatch classification:")
    for text, cat in zip(texts_en, categories):
        print(f"  {cat}: {text[:50]}...")

    # Test top-3 categories
    category_top3 = classifier.classify(
        "Ancient monastery in beautiful natural setting",
        language='en',
        top_k=3
    )
    print(f"\nTop-3 categories: {category_top3}")

    # Statistics
    print("\nStatistics:")
    print(classifier.get_statistics())
    print("\nCategory distribution:")
    print(classifier.get_category_distribution())