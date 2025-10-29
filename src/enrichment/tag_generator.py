"""
Tag Generator
Module for generating relevant tags for Georgian attractions.
Generates descriptive tags like: fortress, waterfall, monastery, museum, etc.
Uses zero-shot classification to match text with predefined tag vocabulary.

Supports:
- English tags: fortress, church, waterfall, lake, mountain, museum, etc.
- Russian tags: крепость, церковь, водопад, озеро, гора, музей, etc.

Uses: joeddav/xlm-roberta-large-xnli model for multilingual tag matching
"""

import logging
from typing import Dict, List, Any, Optional, Union
from transformers import pipeline
import pandas as pd
from datasets import Dataset

logger = logging.getLogger(__name__)


class TagGenerator:
    """
    Tag generator using zero-shot classification.

    Attributes:
        config (dict): Model configuration
        pipeline: Loaded classification pipeline
        tag_vocabulary (dict): Tag lists for each language
        top_k (int): Number of top tags to return
        stats (dict): Generation statistics
    """

    def __init__(self, config: dict):
        """
        Initialize tag generator.

        Args:
            config: Dictionary with model and tag configuration
                Example:
                {
                    'zero_shot': {
                        'model': 'joeddav/xlm-roberta-large-xnli',
                        'device': 0,
                        'batch_size': 32
                    },
                    'tags': {
                        'en': ['fortress', 'church', 'waterfall', 'lake', ...],
                        'ru': ['крепость', 'церковь', 'водопад', 'озеро', ...],
                        'top_k': 5
                    }
                }
        """
        self.config = config
        self.pipeline = None
        self.tag_vocabulary = config.get('tags', {})
        self.top_k = self.tag_vocabulary.get('top_k', 5)
        self.stats = {
            'processed': 0,
            'errors': 0,
            'tag_counts': {}
        }

        logger.info("Initializing tag generator")
        self._load_model()

    def _load_model(self):
        """Load zero-shot classification model."""
        try:
            logger.info("Loading zero-shot classification model for tags...")

            zero_shot_config = self.config.get('zero_shot', {})

            self.pipeline = pipeline(
                "zero-shot-classification",
                model=zero_shot_config.get('model', 'joeddav/xlm-roberta-large-xnli'),
                device=zero_shot_config.get('device', -1)
            )

            self.batch_size = zero_shot_config.get('batch_size', 32)
            self.hypothesis_template = zero_shot_config.get(
                'hypothesis_template',
                'This example is about {}.'
            )

            logger.info(" Tag generation model loaded")

        except Exception as e:
            logger.error(f"Error loading tag generation model: {e}")
            raise

    def generate_tags(
        self,
        text: str,
        language: str = 'en',
        top_k: Optional[int] = None
    ) -> List[str]:
        """
        Generate relevant tags for text.

        Args:
            text: Input text to generate tags for
            language: Language of text ('en' or 'ru')
            top_k: Number of top tags to return (default: from config)

        Returns:
            List of relevant tags sorted by relevance

        Example:
            >>> generator = TagGenerator(config)
            >>> tags = generator.generate_tags(
            ...     "Ancient stone fortress on the hill with great views",
            ...     language='en'
            ... )
            >>> print(tags)
            ['fortress', 'castle', 'historical', 'mountain', 'architecture']
        """
        if not text or not isinstance(text, str):
            logger.warning("Empty or invalid text provided")
            return []

        if language not in self.tag_vocabulary:
            logger.warning(f"Language {language} not supported for tags")
            return []

        if top_k is None:
            top_k = self.top_k

        try:
            candidate_tags = self.tag_vocabulary[language]

            # Ensure we have tags to work with
            if not candidate_tags:
                logger.warning(f"No candidate tags found for language {language}")
                return []

            result = self.pipeline(
                text,
                candidate_labels=candidate_tags,
                hypothesis_template=self.hypothesis_template
            )

            self.stats['processed'] += 1

            # Get top-k tags
            top_tags = result['labels'][:top_k]

            # Update tag counts
            for tag in top_tags:
                self.stats['tag_counts'][tag] = \
                    self.stats['tag_counts'].get(tag, 0) + 1

            return top_tags

        except Exception as e:
            logger.error(f"Error generating tags: {e}")
            self.stats['errors'] += 1
            return []

    def generate_tags_batch(
        self,
        texts: List[str],
        language: str = 'en',
        top_k: Optional[int] = None
    ) -> List[List[str]]:
        """
        Generate tags for multiple texts in batch.

        Args:
            texts: List of texts to generate tags for
            language: Language of texts
            top_k: Number of top tags per text

        Returns:
            List of tag lists for each text

        Example:
            >>> texts = [
            ...     "Beautiful waterfall in the mountains",
            ...     "Medieval church with frescoes"
            ... ]
            >>> tags_batch = generator.generate_tags_batch(texts)
            >>> print(tags_batch[0])
            ['waterfall', 'nature', 'mountain', 'beautiful', 'landscape']
        """
        logger.info(f"Generating tags for batch of {len(texts)} texts...")

        results = []
        for idx, text in enumerate(texts):
            if idx % 100 == 0 and idx > 0:
                logger.info(f"Processed {idx}/{len(texts)} texts")

            tags = self.generate_tags(text, language, top_k)
            results.append(tags)

        logger.info(f" Batch tag generation completed")
        return results

    def enrich_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str,
        language: str = 'en',
        output_column: str = 'tags',
        top_k: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Enrich dataframe with generated tags.

        Args:
            df: Input dataframe
            text_column: Name of column with text
            language: Language of texts
            output_column: Name of output column for tags
            top_k: Number of top tags per text

        Returns:
            Dataframe with added tags column

        Example:
            >>> df = pd.DataFrame({
            ...     'name': ['Narikala', 'Martvili Canyon'],
            ...     'description': [
            ...         'Ancient fortress overlooking the city',
            ...         'Beautiful canyon with turquoise water'
            ...     ]
            ... })
            >>> enriched_df = generator.enrich_dataframe(
            ...     df,
            ...     text_column='description',
            ...     language='en'
            ... )
            >>> print(enriched_df['tags'].tolist())
            [['fortress', 'ancient', 'castle', 'historical', 'architecture'],
             ['canyon', 'nature', 'water', 'beautiful', 'landscape']]
        """
        logger.info(f"Enriching dataframe with tags ({len(df)} records)...")

        if text_column not in df.columns:
            raise ValueError(f"Column {text_column} not found in dataframe")

        # Generate tags for each text
        tags_list = self.generate_tags_batch(
            texts=df[text_column].tolist(),
            language=language,
            top_k=top_k
        )

        # Add to dataframe
        df[output_column] = tags_list

        logger.info(f"✓ Dataframe tag enrichment completed")
        return df

    def generate_tags_dataset(
        self,
        dataset: Union[Dataset, pd.DataFrame],
        column: str,
        language: str = 'en',
        top_k: Optional[int] = None
    ) -> List[List[str]]:
        """
        Generate tags for HuggingFace Dataset or DataFrame column.

        Args:
            dataset: HuggingFace Dataset or pandas DataFrame
            column: Name of text column
            language: Language of texts
            top_k: Number of top tags per text

        Returns:
            List of tag lists for entire dataset

        Example:
            >>> from datasets import load_dataset
            >>> dataset = load_dataset('AIAnastasia/Georgian-attractions')
            >>> tags = generator.generate_tags_dataset(
            ...     dataset['train'],
            ...     column='description_en',
            ...     language='en'
            ... )
        """
        logger.info(f"Generating tags for dataset column '{column}'...")

        # Convert to list of texts
        if isinstance(dataset, Dataset):
            texts = dataset[column]
        elif isinstance(dataset, pd.DataFrame):
            texts = dataset[column].tolist()
        else:
            raise TypeError("Dataset must be HuggingFace Dataset or pandas DataFrame")

        # Generate tags in batches
        return self.generate_tags_batch(texts, language, top_k)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get tag generation statistics.

        Returns:
            Statistics dictionary with tag usage:
            {
                'processed': 500,
                'errors': 5,
                'tag_counts': {
                    'fortress': 45,
                    'church': 120,
                    'waterfall': 30,
                    ...
                },
                'unique_tags': 15,
                'success_rate': 0.99
            }
        """
        success_rate = (
            (self.stats['processed'] - self.stats['errors']) / self.stats['processed']
            if self.stats['processed'] > 0 else 0
        )

        return {
            **self.stats,
            'unique_tags': len(self.stats['tag_counts']),
            'success_rate': round(success_rate, 3)
        }

    def get_top_tags(self, n: int = 10) -> List[tuple]:
        """
        Get most frequently used tags.

        Args:
            n: Number of top tags to return

        Returns:
            List of (tag, count) tuples sorted by frequency

        Example:
            >>> top_tags = generator.get_top_tags(5)
            >>> print(top_tags)
            [('church', 120), ('fortress', 85), ('monastery', 65), ...]
        """
        sorted_tags = sorted(
            self.stats['tag_counts'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_tags[:n]

    def get_tag_distribution(self) -> Dict[str, float]:
        """
        Get percentage distribution of tags.

        Returns:
            Dictionary with tag percentages:
            {
                'church': 24.0,
                'fortress': 17.0,
                'monastery': 13.0,
                ...
            }
        """
        total = sum(self.stats['tag_counts'].values())
        if total == 0:
            return {}

        distribution = {
            tag: round((count / total) * 100, 2)
            for tag, count in self.stats['tag_counts'].items()
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
        'tags': {
            'en': [
                'fortress', 'bridge', 'church', 'waterfall', 'lake',
                'monastery', 'mountain', 'museum', 'river',
                'national park', 'gallery', 'beach', 'park',
                'castle', 'garden', 'reservoir', 'cave', 'canyon', 'forest'
            ],
            'ru': [
                'крепость', 'мост', 'церковь', 'водопад', 'озеро',
                'монастырь', 'гора', 'музей', 'река',
                'национальный парк', 'галерея', 'пляж', 'парк',
                'замок', 'сад', 'водохранилище', 'пещера', 'каньон', 'лес'
            ],
            'top_k': 5
        }
    }

    # Initialize generator
    generator = TagGenerator(config)

    # Test single tag generation - English
    text_en = "Ancient stone fortress on the mountain with beautiful panoramic views"
    tags_en = generator.generate_tags(text_en, language='en')
    print("\nEnglish tags:")
    print(f"Text: {text_en}")
    print(f"Tags: {tags_en}")

    # Test single tag generation - Russian
    text_ru = "Древний монастырь в горах с красивыми фресками"
    tags_ru = generator.generate_tags(text_ru, language='ru')
    print("\nRussian tags:")
    print(f"Text: {text_ru}")
    print(f"Tags: {tags_ru}")

    # Test batch generation
    texts_en = [
        "Beautiful waterfall cascading down the rocks",
        "Medieval church with ancient frescoes",
        "Natural cave system with stalactites"
    ]
    tags_batch = generator.generate_tags_batch(texts_en, language='en')
    print("\nBatch tag generation:")
    for text, tags in zip(texts_en, tags_batch):
        print(f"  {text[:50]}...")
        print(f"  Tags: {tags}")

    # Test top-3 tags only
    tags_top3 = generator.generate_tags(
        "Ancient fortress and monastery complex",
        language='en',
        top_k=3
    )
    print(f"\nTop-3 tags: {tags_top3}")

    # Statistics
    print("\nStatistics:")
    print(generator.get_statistics())
    print("\nTop 5 tags:")
    print(generator.get_top_tags(5))
    print("\nTag distribution:")
    distribution = generator.get_tag_distribution()
    for tag, pct in list(distribution.items())[:5]:
        print(f"  {tag}: {pct}%")