"""
NER (Named Entity Recognition) Processor
Module for extracting named entities from Georgian attraction texts.

Supports:
- English language (model: dslim/bert-base-NER)
- Russian language (model: DeepPavlov/bert-base-cased-conversational)

Extracted entity types:
- LOC (Location) - geographical names
- ORG (Organization) - organizations, museums
- PER (Person) - person names
- MISC (Miscellaneous) - other entities
"""

import logging
from typing import Dict, List, Any, Optional
from transformers import pipeline
import pandas as pd

logger = logging.getLogger(__name__)


class NERProcessor:
    """
    Named Entity Recognition processor for multilingual texts.

    Attributes:
        config (dict): Model configuration
        pipelines (dict): Loaded NER pipelines for each language
        stats (dict): Processing statistics
    """

    def __init__(self, config: dict):
        """
        Initialize NER processor.

        Args:
            config: Dictionary with model configuration
                Example:
                {
                    'ner_en': {
                        'model': 'dslim/bert-base-NER',
                        'aggregation_strategy': 'simple',
                        'device': 0
                    },
                    'ner_ru': {
                        'model': 'DeepPavlov/bert-base-cased-conversational',
                        'aggregation_strategy': 'simple',
                        'device': 0
                    }
                }
        """
        self.config = config
        self.pipelines = {}
        self.stats = {
            'processed': 0,
            'with_entities': 0,
            'errors': 0
        }

        logger.info("Initializing NER processor")
        self._load_models()

    def _load_models(self):
        """Load NER models for all languages."""
        try:
            # Load English model
            logger.info("Loading English NER model")
            self.pipelines['en'] = pipeline(
                "ner",
                model=self.config['ner_en']['model'],
                aggregation_strategy=self.config['ner_en']['aggregation_strategy'],
                device=self.config['ner_en'].get('device', -1)
            )
            logger.info("English NER model loaded")

            # Load Russian model
            logger.info("Loading Russian NER model")
            self.pipelines['ru'] = pipeline(
                "ner",
                model=self.config['ner_ru']['model'],
                aggregation_strategy=self.config['ner_ru']['aggregation_strategy'],
                device=self.config['ner_ru'].get('device', -1)
            )
            logger.info("Russian NER model loaded")

        except Exception as e:
            logger.error(f"Error loading NER models: {e}")
            raise

    def extract_entities(
        self,
        text: str,
        language: str = 'en',
        min_score: float = 0.5
    ) -> Dict[str, List[str]]:
        """
        Extract named entities from text.

        Args:
            text: Input text for analysis
            language: Text language ('en' or 'ru')
            min_score: Minimum confidence threshold (0-1)

        Returns:
            Dictionary with entity types:
            {
                'locations': ['Tbilisi', 'Georgia'],
                'organizations': ['National Museum'],
                'persons': [],
                'misc': ['Caucasus']
            }

        Example:
            >>> processor = NERProcessor(config)
            >>> entities = processor.extract_entities(
            ...     "Narikala Fortress is located in Tbilisi",
            ...     language='en'
            ... )
            >>> print(entities['locations'])
            ['Narikala', 'Tbilisi']
        """
        if not text or not isinstance(text, str):
            return self._empty_entities()

        if language not in self.pipelines:
            logger.warning(f"Language {language} is not supported")
            return self._empty_entities()

        try:
            # Extract entities via pipeline
            ner_results = self.pipelines[language](text)

            # Group by types
            entities = {
                'locations': [],
                'organizations': [],
                'persons': [],
                'misc': []
            }

            for entity in ner_results:
                if entity['score'] < min_score:
                    continue

                entity_text = entity['word'].strip()
                entity_type = entity['entity_group']

                # Map entity types
                if entity_type in ['LOC', 'LOCATION']:
                    entities['locations'].append(entity_text)
                elif entity_type in ['ORG', 'ORGANIZATION']:
                    entities['organizations'].append(entity_text)
                elif entity_type in ['PER', 'PERSON']:
                    entities['persons'].append(entity_text)
                else:
                    entities['misc'].append(entity_text)

            # Remove duplicates while preserving order
            for key in entities:
                entities[key] = list(dict.fromkeys(entities[key]))

            self.stats['processed'] += 1
            if any(entities.values()):
                self.stats['with_entities'] += 1

            return entities

        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            self.stats['errors'] += 1
            return self._empty_entities()

    def process_batch(
        self,
        texts: List[str],
        language: str = 'en',
        min_score: float = 0.5
    ) -> List[Dict[str, List[str]]]:
        """
        Batch processing of text list.

        Args:
            texts: List of texts to process
            language: Language of texts
            min_score: Minimum confidence threshold

        Returns:
            List of entity dictionaries for each text
        """
        results = []
        for text in texts:
            entities = self.extract_entities(text, language, min_score)
            results.append(entities)

        return results

    def enrich_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str,
        language: str = 'en',
        output_column: str = 'ner_entities'
    ) -> pd.DataFrame:
        """
        Enrich dataframe with entities.

        Args:
            df: Input dataframe
            text_column: Name of text column
            language: Language of texts
            output_column: Name of output column for results

        Returns:
            Dataframe with added entities column

        Example:
            >>> df = pd.DataFrame({
            ...     'name': ['Narikala'],
            ...     'description': ['Ancient fortress in Tbilisi']
            ... })
            >>> enriched_df = processor.enrich_dataframe(
            ...     df,
            ...     text_column='description',
            ...     language='en'
            ... )
        """
        logger.info(f"Enriching dataframe ({len(df)} records)...")

        if text_column not in df.columns:
            raise ValueError(f"Column {text_column} not found in dataframe")

        # Extract entities for each row
        entities_list = []
        for idx, text in enumerate(df[text_column]):
            if idx % 100 == 0 and idx > 0:
                logger.info(f"Processed {idx}/{len(df)} records")

            entities = self.extract_entities(text, language)
            entities_list.append(entities)

        # Add results column
        df[output_column] = entities_list

        logger.info(f"✓ Enrichment completed. Entities extracted in {self.stats['with_entities']} records")

        return df

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get processing statistics.

        Returns:
            Statistics dictionary:
            {
                'processed': 150,
                'with_entities': 142,
                'errors': 8,
                'success_rate': 0.95
            }
        """
        success_rate = (
            self.stats['with_entities'] / self.stats['processed']
            if self.stats['processed'] > 0 else 0
        )

        return {
            **self.stats,
            'success_rate': round(success_rate, 3)
        }

    @staticmethod
    def _empty_entities() -> Dict[str, List[str]]:
        """Return empty entities structure."""
        return {
            'locations': [],
            'organizations': [],
            'persons': [],
            'misc': []
        }


# Usage example
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Example configuration
    config = {
        'ner_en': {
            'model': 'dslim/bert-base-NER',
            'aggregation_strategy': 'simple',
            'device': -1  # CPU
        },
        'ner_ru': {
            'model': 'DeepPavlov/bert-base-cased-conversational',
            'aggregation_strategy': 'simple',
            'device': -1
        }
    }

    # Initialize processor
    processor = NERProcessor(config)

    # Test in English
    text_en = "Narikala Fortress is located in Tbilisi, Georgia."
    entities_en = processor.extract_entities(text_en, language='en')
    print("\nEnglish text:")
    print(f"Text: {text_en}")
    print(f"Entities: {entities_en}")

    # Test in Russian
    text_ru = "Крепость Нарикала расположена в Тбилиси, столице Грузии."
    entities_ru = processor.extract_entities(text_ru, language='ru')
    print("\nRussian text:")
    print(f"Text: {text_ru}")
    print(f"Entities: {entities_ru}")

    # Statistics
    print("\nStatistics:")
    print(processor.get_statistics())