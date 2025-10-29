"""
Enrichment Pipeline
===================

Main pipeline for enriching Georgian attractions dataset.

Orchestrates:
1. Data loading (HuggingFace or local files)
2. NER extraction (locations, organizations, etc.)
3. Category classification (nature, history, culture, spirituality, architecture)
4. Tag generation (fortress, waterfall, monastery, etc.)
5. Fuzzy matching (Russian ↔ English name matching)
6. Data validation and saving

Features:
- Batch processing with progress tracking
- Error handling and recovery
- Statistics and quality reports
- Configurable pipeline steps
- Resume from checkpoint support
"""

import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import yaml

from src.enrichment.ner_processor import NERProcessor
from src.enrichment.category_classifier import CategoryClassifier
from src.enrichment.tag_generator import TagGenerator
from src.utils.fuzzy_matching import FuzzyMatcher
from src.utils.data_loader import DataLoader

logger = logging.getLogger(__name__)


class EnrichmentPipeline:
    """
    Main enrichment pipeline for Georgian attractions.

    Attributes:
        config (dict): Pipeline configuration
        data_loader (DataLoader): Data loading component
        ner_processor (NERProcessor): NER extraction component
        category_classifier (CategoryClassifier): Category classification component
        tag_generator (TagGenerator): Tag generation component
        fuzzy_matcher (FuzzyMatcher): Fuzzy matching component
        stats (dict): Pipeline statistics
    """

    def __init__(self, config_path: Optional[str] = None, config: Optional[dict] = None):
        """
        Initialize enrichment pipeline.

        Args:
            config_path: Path to YAML configuration file
            config: Configuration dictionary (overrides config_path)

        Example:
            >>> # From config file
            >>> pipeline = EnrichmentPipeline('config/config.yaml')
            >>> # From dict
            >>> pipeline = EnrichmentPipeline(config={...})
        """
        # Load configuration
        if config:
            self.config = config
        elif config_path:
            self.config = self._load_config(config_path)
        else:
            raise ValueError("Either config_path or config must be provided")

        self.stats = {
            'total_records': 0,
            'enriched_records': 0,
            'errors': 0,
            'processing_time': 0
        }

        logger.info("=" * 60)
        logger.info("Initializing Enrichment Pipeline")
        logger.info("=" * 60)

        # Initialize components
        self._initialize_components()

        logger.info(" Pipeline initialized successfully")

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        logger.info(f"Loading configuration from: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        logger.info(" Configuration loaded")
        return config

    def _initialize_components(self):
        """Initialize all pipeline components."""
        logger.info("Initializing pipeline components...")

        # Data Loader
        logger.info("  [1/5] Initializing Data Loader...")
        self.data_loader = DataLoader(self.config)

        # NER Processor
        logger.info("  [2/5] Initializing NER Processor...")
        self.ner_processor = NERProcessor(self.config)

        # Category Classifier
        logger.info("  [3/5] Initializing Category Classifier...")
        self.category_classifier = CategoryClassifier(self.config)

        # Tag Generator
        logger.info("  [4/5] Initializing Tag Generator...")
        self.tag_generator = TagGenerator(self.config)

        # Fuzzy Matcher
        logger.info("  [5/5] Initializing Fuzzy Matcher...")
        self.fuzzy_matcher = FuzzyMatcher(self.config)

        logger.info(" All components initialized")

    def run(
        self,
        source: Optional[str] = None,
        output_path: Optional[str] = None,
        steps: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Run complete enrichment pipeline.

        Args:
            source: Data source (overrides config)
            output_path: Output file path (overrides config)
            steps: List of steps to run (default: all)
                   Options: ['ner', 'category', 'tags', 'fuzzy_match']

        Returns:
            Enriched DataFrame

        Example:
            >>> pipeline = EnrichmentPipeline('config/config.yaml')
            >>> enriched_df = pipeline.run()
            >>> # Or with custom source
            >>> enriched_df = pipeline.run(source='./data/attractions.csv')
        """
        logger.info("\n" + "=" * 60)
        logger.info("STARTING ENRICHMENT PIPELINE")
        logger.info("=" * 60 + "\n")

        import time
        start_time = time.time()

        # Default: run all steps
        if steps is None:
            steps = ['ner', 'category', 'tags', 'fuzzy_match']

        try:
            # Step 1: Load data
            logger.info("\n[STEP 1/5] Loading Data")
            logger.info("-" * 60)
            df = self._load_data(source)
            self.stats['total_records'] = len(df)

            # Step 2: Enrich Russian data
            logger.info("\n[STEP 2/5] Enriching Russian Data")
            logger.info("-" * 60)
            df_ru = self._enrich_language_data(df, language='ru', steps=steps)

            # Step 3: Enrich English data
            logger.info("\n[STEP 3/5] Enriching English Data")
            logger.info("-" * 60)
            df_en = self._enrich_language_data(df, language='en', steps=steps)

            # Step 4: Fuzzy matching (if enabled)
            if 'fuzzy_match' in steps:
                logger.info("\n[STEP 4/5] Fuzzy Matching RU ↔ EN")
                logger.info("-" * 60)
                matches_df = self._fuzzy_match(df_ru, df_en)
            else:
                logger.info("\n[STEP 4/5] Fuzzy Matching - SKIPPED")
                matches_df = None

            # Step 5: Combine and save results
            logger.info("\n[STEP 5/5] Saving Results")
            logger.info("-" * 60)
            result_df = self._save_results(df_ru, df_en, matches_df, output_path)

            self.stats['enriched_records'] = len(result_df)
            self.stats['processing_time'] = round(time.time() - start_time, 2)

            # Print final statistics
            self._print_final_statistics()

            logger.info("\n" + "=" * 60)
            logger.info(" PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 60 + "\n")

            return result_df

        except Exception as e:
            logger.error(f"\n Pipeline failed: {e}")
            self.stats['errors'] += 1
            raise

    def _load_data(self, source: Optional[str] = None) -> pd.DataFrame:
        """Load data using DataLoader."""
        df = self.data_loader.load(source=source)
        logger.info(f"✓ Loaded {len(df)} records")

        return df

    def _enrich_language_data(
        self,
        df: pd.DataFrame,
        language: str,
        steps: List[str]
    ) -> pd.DataFrame:
        """
        Enrich data for specific language.

        Args:
            df: Input dataframe
            language: Language code ('ru' or 'en')
            steps: Pipeline steps to execute

        Returns:
            Enriched dataframe for this language
        """
        # Determine column names based on language
        text_column = f'description_{language}'

        if text_column not in df.columns:
            logger.warning(f"Column {text_column} not found, skipping {language}")
            return df

        # Create a copy for this language
        df_lang = df.copy()

        # NER extraction
        if 'ner' in steps:
            logger.info(f"  Extracting NER entities ({language})...")
            df_lang = self.ner_processor.enrich_dataframe(
                df_lang,
                text_column=text_column,
                language=language,
                output_column=f'ner_{language}'
            )
            logger.info(f"   NER extraction completed")

        # Category classification
        if 'category' in steps:
            logger.info(f"  Classifying categories ({language})...")
            df_lang = self.category_classifier.enrich_dataframe(
                df_lang,
                text_column=text_column,
                language=language,
                output_column=f'category_{language}'
            )
            logger.info(f"   Category classification completed")

        # Tag generation
        if 'tags' in steps:
            logger.info(f"  Generating tags ({language})...")
            df_lang = self.tag_generator.enrich_dataframe(
                df_lang,
                text_column=text_column,
                language=language,
                output_column=f'tags_{language}'
            )
            logger.info(f"   Tag generation completed")

        return df_lang

    def _fuzzy_match(
        self,
        df_ru: pd.DataFrame,
        df_en: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Perform fuzzy matching between Russian and English names.

        Args:
            df_ru: Russian dataframe
            df_en: English dataframe

        Returns:
            DataFrame with matched pairs
        """
        name_col_ru = 'name_ru'
        name_col_en = 'name_en'

        # Check if name columns exist
        if name_col_ru not in df_ru.columns or name_col_en not in df_en.columns:
            logger.warning("Name columns not found, skipping fuzzy matching")
            return pd.DataFrame()

        logger.info(f"  Matching {len(df_ru)} RU names with {len(df_en)} EN names...")

        matches_df = self.fuzzy_matcher.match_dataframes(
            df_ru, df_en,
            source_column=name_col_ru,
            target_column=name_col_en,
            source_name='ru',
            target_name='en'
        )

        logger.info(f"   Found {len(matches_df)} matches")

        # Generate quality report
        if not matches_df.empty:
            quality_report = self.fuzzy_matcher.get_match_quality_report(matches_df)
            logger.info(f"  Match quality:")
            logger.info(f"    Average score: {quality_report.get('average_score', 0)}")
            logger.info(f"    High confidence: {quality_report.get('confidence_distribution', {}).get('high', 0)}")

        return matches_df

    def _save_results(
        self,
        df_ru: pd.DataFrame,
        df_en: pd.DataFrame,
        matches_df: Optional[pd.DataFrame],
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Save enrichment results.

        Args:
            df_ru: Enriched Russian dataframe
            df_en: Enriched English dataframe
            matches_df: Fuzzy matching results
            output_path: Custom output path

        Returns:
            Combined results dataframe
        """
        output_dir = self.config.get('data', {}).get('output_dir', './data/processed')
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Save Russian data
        ru_path = f"{output_dir}/attractions_ru_enriched.csv"
        logger.info(f"  Saving Russian data to: {ru_path}")
        df_ru.to_csv(ru_path, index=False)

        # Save English data
        en_path = f"{output_dir}/attractions_en_enriched.csv"
        logger.info(f"  Saving English data to: {en_path}")
        df_en.to_csv(en_path, index=False)

        # Save fuzzy matches if available
        if matches_df is not None and not matches_df.empty:
            matches_path = f"{output_dir}/name_matches.csv"
            logger.info(f"  Saving fuzzy matches to: {matches_path}")
            matches_df.to_csv(matches_path, index=False)

        # Save combined data if requested
        if output_path:
            logger.info(f"  Saving combined data to: {output_path}")
            # Merge on common ID if exists, otherwise concatenate
            if 'id' in df_ru.columns and 'id' in df_en.columns:
                combined_df = pd.merge(df_ru, df_en, on='id', suffixes=('_ru', '_en'))
            else:
                combined_df = pd.concat([df_ru, df_en], axis=0, ignore_index=True)

            combined_df.to_csv(output_path, index=False)
            logger.info(f"   Saved combined results")
            return combined_df

        logger.info(f"   All results saved to: {output_dir}")
        return df_ru  # Return Russian as default

    def _print_final_statistics(self):
        """Print comprehensive pipeline statistics."""
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE STATISTICS")
        logger.info("=" * 60)

        # Overall stats
        logger.info(f"\n Overall:")
        logger.info(f"  Total records processed: {self.stats['total_records']}")
        logger.info(f"  Enriched records: {self.stats['enriched_records']}")
        logger.info(f"  Errors: {self.stats['errors']}")
        logger.info(f"  Processing time: {self.stats['processing_time']}s")

        # Component stats
        logger.info(f"\n NER Processor:")
        ner_stats = self.ner_processor.get_statistics()
        for key, value in ner_stats.items():
            logger.info(f"  {key}: {value}")

        logger.info(f"\n  Category Classifier:")
        cat_stats = self.category_classifier.get_statistics()
        for key, value in cat_stats.items():
            logger.info(f"  {key}: {value}")

        logger.info(f"\n  Tag Generator:")
        tag_stats = self.tag_generator.get_statistics()
        logger.info(f"  Processed: {tag_stats['processed']}")
        logger.info(f"  Unique tags: {tag_stats['unique_tags']}")
        logger.info(f"  Top tags: {self.tag_generator.get_top_tags(3)}")

        logger.info(f"\n Fuzzy Matcher:")
        match_stats = self.fuzzy_matcher.get_statistics()
        for key, value in match_stats.items():
            logger.info(f"  {key}: {value}")

        logger.info("=" * 60 + "\n")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive pipeline statistics.

        Returns:
            Dictionary with all component statistics
        """
        return {
            'pipeline': self.stats,
            'data_loader': self.data_loader.get_statistics(),
            'ner_processor': self.ner_processor.get_statistics(),
            'category_classifier': self.category_classifier.get_statistics(),
            'tag_generator': self.tag_generator.get_statistics(),
            'fuzzy_matcher': self.fuzzy_matcher.get_statistics()
        }


# Usage example
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Example configuration
    config = {
        'data': {
            'source': 'AIAnastasia/Georgian-attractions',
            'cache_dir': './data/cache',
            'output_dir': './data/processed'
        },
        'validation': {
            'required_fields': ['name_ru', 'name_en', 'description_ru', 'description_en'],
            'check_empty_values': True
        },
        'ner_en': {
            'model': 'dslim/bert-base-NER',
            'aggregation_strategy': 'simple',
            'device': -1
        },
        'ner_ru': {
            'model': 'DeepPavlov/bert-base-cased-conversational',
            'aggregation_strategy': 'simple',
            'device': -1
        },
        'zero_shot': {
            'model': 'joeddav/xlm-roberta-large-xnli',
            'device': -1,
            'batch_size': 32,
            'hypothesis_template': 'This example is about {}.'
        },
        'categories': {
            'en': ['nature', 'history', 'culture', 'spirituality', 'architecture'],
            'ru': ['природа', 'история', 'культура', 'духовность', 'архитектура']
        },
        'tags': {
            'en': ['fortress', 'church', 'waterfall', 'lake', 'monastery', 'mountain'],
            'ru': ['крепость', 'церковь', 'водопад', 'озеро', 'монастырь', 'гора'],
            'top_k': 5
        },
        'fuzzy_threshold': 65,
        'weights': {
            'ratio': 0.4,
            'token_set': 0.5,
            'partial': 0.1
        }
    }

    # Initialize and run pipeline
    print("Initializing pipeline...")
    pipeline = EnrichmentPipeline(config=config)

    print("\nRunning enrichment pipeline...")
    enriched_df = pipeline.run()

    print(f"\n✓ Pipeline completed!")
    print(f"Enriched {len(enriched_df)} records")

    # Get statistics
    stats = pipeline.get_statistics()
    print("\nFinal statistics:")
    print(stats)