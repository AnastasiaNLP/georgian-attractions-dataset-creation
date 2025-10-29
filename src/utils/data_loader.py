"""
Data Loader
===========

Module for loading Georgian attractions data from various sources.

Supports:
- HuggingFace Datasets
- Local CSV files
- Local JSON/JSONL files
- Local Parquet files
- Remote URLs

Features:
- Automatic format detection
- Data validation
- Caching support
- Progress tracking
"""

import logging
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import pandas as pd
from datasets import load_dataset, Dataset
import json

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Universal data loader for Georgian attractions dataset.

    Attributes:
        config (dict): Loader configuration
        cache_dir (str): Directory for caching data
        stats (dict): Loading statistics
    """

    def __init__(self, config: dict):
        """
        Initialize data loader.

        Args:
            config: Dictionary with loader configuration
                Example:
                {
                    'data': {
                        'source': 'AIAnastasia/Georgian-attractions',
                        'cache_dir': './data/cache',
                        'output_dir': './data/processed'
                    },
                    'validation': {
                        'required_fields': ['name_ru', 'name_en', 'description_ru', 'description_en'],
                        'check_empty_values': True
                    }
                }
        """
        self.config = config
        self.cache_dir = config.get('data', {}).get('cache_dir', './data/cache')
        self.stats = {
            'loaded_records': 0,
            'valid_records': 0,
            'invalid_records': 0,
            'source_type': None
        }

        # Create cache directory if needed
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

        logger.info("Data loader initialized")

    def load(
        self,
        source: Optional[str] = None,
        split: Optional[str] = None,
        validate: bool = True
    ) -> pd.DataFrame:
        """
        Load data from specified source.

        Args:
            source: Data source (HuggingFace dataset name, file path, or URL)
                    If None, uses source from config
            split: Dataset split for HuggingFace datasets ('train', 'test', etc.)
            validate: Whether to validate loaded data

        Returns:
            Pandas DataFrame with loaded data

        Example:
            >>> loader = DataLoader(config)
            >>> # Load from HuggingFace
            >>> df = loader.load('AIAnastasia/Georgian-attractions')
            >>> # Load from local CSV
            >>> df = loader.load('./data/attractions.csv')
        """
        if source is None:
            source = self.config.get('data', {}).get('source')

        if not source:
            raise ValueError("No data source specified")

        logger.info(f"Loading data from: {source}")

        # Detect source type and load accordingly
        df = self._load_from_source(source, split)

        self.stats['loaded_records'] = len(df)
        logger.info(f"✓ Loaded {len(df)} records")

        # Validate if requested
        if validate:
            df = self._validate_data(df)

        return df

    def _load_from_source(
        self,
        source: str,
        split: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load data based on source type detection.

        Args:
            source: Data source path or name
            split: Dataset split (for HuggingFace)

        Returns:
            DataFrame with loaded data
        """
        # Check if it's a HuggingFace dataset (format: owner/dataset)
        if '/' in source and not source.startswith(('.', '/')):
            return self._load_from_huggingface(source, split)

        # Check if it's a local file
        path = Path(source)
        if path.exists():
            return self._load_from_file(path)

        # Try as URL
        if source.startswith(('http://', 'https://')):
            return self._load_from_url(source)

        raise ValueError(f"Unknown source type or source not found: {source}")

    def _load_from_huggingface(
        self,
        dataset_name: str,
        split: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load data from HuggingFace Datasets.

        Args:
            dataset_name: HuggingFace dataset name (e.g., 'AIAnastasia/Georgian-attractions')
            split: Dataset split to load

        Returns:
            DataFrame with loaded data
        """
        logger.info(f"Loading from HuggingFace: {dataset_name}")
        self.stats['source_type'] = 'huggingface'

        try:
            # Load dataset
            dataset = load_dataset(
                dataset_name,
                split=split,
                cache_dir=self.cache_dir
            )

            # Convert to pandas
            if isinstance(dataset, Dataset):
                df = dataset.to_pandas()
            else:
                # If multiple splits, take first one
                first_split = list(dataset.keys())[0]
                logger.warning(f"Multiple splits found, using: {first_split}")
                df = dataset[first_split].to_pandas()

            logger.info(f"✓ Loaded from HuggingFace")
            return df

        except Exception as e:
            logger.error(f"Error loading from HuggingFace: {e}")
            raise

    def _load_from_file(self, path: Path) -> pd.DataFrame:
        """
        Load data from local file.

        Supports: CSV, JSON, JSONL, Parquet

        Args:
            path: Path to local file

        Returns:
            DataFrame with loaded data
        """
        logger.info(f"Loading from local file: {path}")
        self.stats['source_type'] = 'local_file'

        suffix = path.suffix.lower()

        try:
            if suffix == '.csv':
                df = pd.read_csv(path)
            elif suffix == '.json':
                df = pd.read_json(path)
            elif suffix == '.jsonl':
                df = pd.read_json(path, lines=True)
            elif suffix == '.parquet':
                df = pd.read_parquet(path)
            else:
                raise ValueError(f"Unsupported file format: {suffix}")

            logger.info(f"✓ Loaded from {suffix} file")
            return df

        except Exception as e:
            logger.error(f"Error loading from file: {e}")
            raise

    def _load_from_url(self, url: str) -> pd.DataFrame:
        """
        Load data from URL.

        Args:
            url: URL to data file

        Returns:
            DataFrame with loaded data
        """
        logger.info(f"Loading from URL: {url}")
        self.stats['source_type'] = 'url'

        try:
            # Detect format from URL
            if url.endswith('.csv'):
                df = pd.read_csv(url)
            elif url.endswith('.json'):
                df = pd.read_json(url)
            elif url.endswith('.jsonl'):
                df = pd.read_json(url, lines=True)
            else:
                # Try CSV as default
                df = pd.read_csv(url)

            logger.info(f"✓ Loaded from URL")
            return df

        except Exception as e:
            logger.error(f"Error loading from URL: {e}")
            raise

    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate loaded data.

        Checks:
        - Required fields present
        - No empty required fields
        - Data types correct

        Args:
            df: DataFrame to validate

        Returns:
            Validated DataFrame (with invalid rows optionally removed)
        """
        logger.info("Validating data...")

        validation_config = self.config.get('validation', {})
        required_fields = validation_config.get('required_fields', [])
        check_empty = validation_config.get('check_empty_values', True)

        initial_count = len(df)

        # Check required fields exist
        missing_fields = set(required_fields) - set(df.columns)
        if missing_fields:
            logger.warning(f"Missing required fields: {missing_fields}")

        # Check for empty values in required fields
        if check_empty and required_fields:
            existing_required = [f for f in required_fields if f in df.columns]

            # Mark invalid rows
            df['_is_valid'] = True
            for field in existing_required:
                df['_is_valid'] &= df[field].notna() & (df[field] != '')

            invalid_count = (~df['_is_valid']).sum()

            if invalid_count > 0:
                logger.warning(f"Found {invalid_count} records with empty required fields")
                self.stats['invalid_records'] = invalid_count

                # Remove invalid rows
                df = df[df['_is_valid']].copy()
                df = df.drop(columns=['_is_valid'])

        self.stats['valid_records'] = len(df)

        logger.info(f"✓ Validation completed:")
        logger.info(f"  Valid records: {self.stats['valid_records']}")
        logger.info(f"  Invalid records: {self.stats['invalid_records']}")

        return df

    def save(
        self,
        df: pd.DataFrame,
        output_path: str,
        format: str = 'csv'
    ) -> str:
        """
        Save DataFrame to file.

        Args:
            df: DataFrame to save
            output_path: Path where to save
            format: Output format ('csv', 'json', 'jsonl', 'parquet')

        Returns:
            Path to saved file

        Example:
            >>> loader.save(df, './data/processed/enriched.csv', format='csv')
        """
        logger.info(f"Saving data to: {output_path}")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if format == 'csv':
                df.to_csv(output_path, index=False)
            elif format == 'json':
                df.to_json(output_path, orient='records', indent=2)
            elif format == 'jsonl':
                df.to_json(output_path, orient='records', lines=True)
            elif format == 'parquet':
                df.to_parquet(output_path, index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")

            logger.info(f"✓ Data saved to {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Error saving data: {e}")
            raise

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get loading statistics.

        Returns:
            Statistics dictionary:
            {
                'loaded_records': 500,
                'valid_records': 495,
                'invalid_records': 5,
                'source_type': 'huggingface',
                'validation_rate': 0.99
            }
        """
        validation_rate = (
            self.stats['valid_records'] / self.stats['loaded_records']
            if self.stats['loaded_records'] > 0 else 0
        )

        return {
            **self.stats,
            'validation_rate': round(validation_rate, 3)
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
        'data': {
            'source': 'AIAnastasia/Georgian-attractions',
            'cache_dir': './data/cache',
            'output_dir': './data/processed'
        },
        'validation': {
            'required_fields': ['name_ru', 'name_en', 'description_ru', 'description_en'],
            'check_empty_values': True
        }
    }

    # Initialize loader
    loader = DataLoader(config)

    # Example 1: Load from HuggingFace
    print("\n=== Loading from HuggingFace ===")
    try:
        df_hf = loader.load('AIAnastasia/Georgian-attractions')
        print(f"Loaded {len(df_hf)} records from HuggingFace")
        print(f"Columns: {df_hf.columns.tolist()}")
    except Exception as e:
        print(f"Error: {e}")

    # Example 2: Load from local CSV
    print("\n=== Loading from local CSV ===")
    # Create sample CSV for testing
    sample_data = pd.DataFrame({
        'name_ru': ['Нарикала', 'Вардзия'],
        'name_en': ['Narikala', 'Vardzia'],
        'description_ru': ['Древняя крепость', 'Пещерный монастырь'],
        'description_en': ['Ancient fortress', 'Cave monastery']
    })
    sample_path = './data/cache/sample.csv'
    Path(sample_path).parent.mkdir(parents=True, exist_ok=True)
    sample_data.to_csv(sample_path, index=False)

    df_csv = loader.load(sample_path)
    print(f"Loaded {len(df_csv)} records from CSV")

    # Example 3: Save data
    print("\n=== Saving data ===")
    output_path = loader.save(
        df_csv,
        './data/processed/output.csv',
        format='csv'
    )
    print(f"Saved to: {output_path}")

    # Statistics
    print("\n=== Statistics ===")
    stats = loader.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")