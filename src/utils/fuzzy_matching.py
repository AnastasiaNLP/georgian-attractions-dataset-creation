"""
Fuzzy Matching Module
=====================

Module for fuzzy matching of Russian and English attraction names.

Matches names like:
- "Нарикала" ↔ "Narikala"
- "Светицховели" ↔ "Svetitskhoveli"
- "Вардзия" ↔ "Vardzia"

Uses combination of three fuzzy matching algorithms:
1. Levenshtein distance (ratio)
2. Token set ratio
3. Partial ratio

Features:
- Text normalization (lowercase, transliteration, punctuation removal)
- Configurable matching thresholds
- Confidence scoring
- Batch matching support
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Any
from unidecode import unidecode
from fuzzywuzzy import fuzz
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


class FuzzyMatcher:
    """
    Fuzzy matcher for bilingual name matching.

    Attributes:
        config (dict): Matching configuration
        threshold (float): Minimum match score threshold
        weights (dict): Weights for different algorithms
        stats (dict): Matching statistics
    """

    def __init__(self, config: dict):
        """
        Initialize fuzzy matcher.

        Args:
            config: Dictionary with matching configuration
                Example:
                {
                    'fuzzy_threshold': 65,
                    'weights': {
                        'ratio': 0.4,
                        'token_set': 0.5,
                        'partial': 0.1
                    },
                    'high_confidence': 90,
                    'manual_review_threshold': 70
                }
        """
        self.config = config
        self.threshold = config.get('fuzzy_threshold', 65)
        self.weights = config.get('weights', {
            'ratio': 0.4,
            'token_set': 0.5,
            'partial': 0.1
        })
        self.high_confidence = config.get('high_confidence', 90)
        self.manual_review_threshold = config.get('manual_review_threshold', 70)

        self.stats = {
            'total_matches': 0,
            'high_confidence_matches': 0,
            'low_confidence_matches': 0,
            'no_matches': 0
        }

        logger.info("Fuzzy matcher initialized")

    @staticmethod
    def normalize_name(name: str) -> str:
        """
        Normalize name for matching.

        Applies:
        1. Convert to lowercase
        2. Remove punctuation
        3. Transliterate Cyrillic to Latin

        Args:
            name: Original name

        Returns:
            Normalized name

        Example:
            >>> FuzzyMatcher.normalize_name("Крепость Нарикала!")
            'krepost narikala'
        """
        if not name or not isinstance(name, str):
            return ""

        # Convert to lowercase and strip
        name = str(name).strip().lower()

        # Remove punctuation
        name = re.sub(r'[^\w\s]', '', name)

        # Transliterate Cyrillic to Latin (грузия -> gruziya)
        name = unidecode(name)

        return name

    def match(
        self,
        source: str,
        target: str
    ) -> float:
        """
        Calculate fuzzy match score between two names.

        Uses weighted combination of three algorithms:
        - ratio: Exact character matching
        - token_set_ratio: Word-based matching
        - partial_ratio: Substring matching

        Args:
            source: Source name
            target: Target name

        Returns:
            Match score from 0 to 100

        Example:
            >>> matcher = FuzzyMatcher(config)
            >>> score = matcher.match("Narikala", "Нарикала")
            >>> print(score)
            87.5
        """
        # Normalize both names
        source_norm = self.normalize_name(source)
        target_norm = self.normalize_name(target)

        if not source_norm or not target_norm:
            return 0.0

        # Calculate three different scores
        ratio_score = fuzz.ratio(source_norm, target_norm)
        token_set_score = fuzz.token_set_ratio(source_norm, target_norm)
        partial_score = fuzz.partial_ratio(source_norm, target_norm)

        # Weighted combination
        combined_score = (
            self.weights['ratio'] * ratio_score +
            self.weights['token_set'] * token_set_score +
            self.weights['partial'] * partial_score
        )

        return round(combined_score, 2)

    def find_best_match(
        self,
        source: str,
        targets: List[str],
        return_score: bool = True
    ) -> Optional[Tuple[str, float]]:
        """
        Find best matching target for source name.

        Args:
            source: Source name to match
            targets: List of target names to match against
            return_score: Whether to return score with match

        Returns:
            If return_score=True: Tuple of (best_match, score)
            If return_score=False: Just best_match string
            Returns None if no match above threshold

        Example:
            >>> targets = ["Narikala", "Vardzia", "Ananuri"]
            >>> match, score = matcher.find_best_match("Нарикала", targets)
            >>> print(f"{match}: {score}")
            Narikala: 87.5
        """
        if not source or not targets:
            return None

        best_match = None
        best_score = 0.0

        for target in targets:
            score = self.match(source, target)

            if score > self.threshold and score > best_score:
                best_score = score
                best_match = target

        if best_match:
            return (best_match, best_score) if return_score else best_match

        return None

    def match_dataframes(
        self,
        df_source: pd.DataFrame,
        df_target: pd.DataFrame,
        source_column: str,
        target_column: str,
        source_name: str = 'source',
        target_name: str = 'target'
    ) -> pd.DataFrame:
        """
        Match names between two dataframes.

        Args:
            df_source: Source dataframe
            df_target: Target dataframe
            source_column: Column name with source names
            target_column: Column name with target names
            source_name: Label for source (e.g., 'ru', 'source')
            target_name: Label for target (e.g., 'en', 'target')

        Returns:
            DataFrame with matched pairs and scores

        Example:
            >>> df_ru = pd.DataFrame({'name_ru': ['Нарикала', 'Вардзия']})
            >>> df_en = pd.DataFrame({'name_en': ['Narikala', 'Vardzia']})
            >>> matches_df = matcher.match_dataframes(
            ...     df_ru, df_en,
            ...     source_column='name_ru',
            ...     target_column='name_en'
            ... )
        """
        logger.info(f"Matching {len(df_source)} source names with {len(df_target)} target names...")

        matches = []

        # Get list of target names
        target_names = df_target[target_column].tolist()

        for idx, row in tqdm(df_source.iterrows(), total=len(df_source), desc="Matching"):
            source_name_val = row[source_column]

            # Find best match
            result = self.find_best_match(source_name_val, target_names, return_score=True)

            if result:
                best_match, score = result

                # Determine confidence level
                if score >= self.high_confidence:
                    confidence = 'high'
                    self.stats['high_confidence_matches'] += 1
                elif score >= self.manual_review_threshold:
                    confidence = 'medium'
                else:
                    confidence = 'low'
                    self.stats['low_confidence_matches'] += 1

                # Get additional columns from source
                match_data = {
                    f'name_{source_name}': source_name_val,
                    f'name_{target_name}': best_match,
                    'match_score': score,
                    'confidence': confidence
                }

                # Add any additional columns you want to preserve
                for col in df_source.columns:
                    if col != source_column and not col.startswith('Unnamed'):
                        match_data[f'{source_name}_{col}'] = row[col]

                matches.append(match_data)
                self.stats['total_matches'] += 1
            else:
                self.stats['no_matches'] += 1

        matches_df = pd.DataFrame(matches)

        logger.info(f"✓ Matching completed:")
        logger.info(f"  Total matches: {self.stats['total_matches']}")
        logger.info(f"  High confidence: {self.stats['high_confidence_matches']}")
        logger.info(f"  Low confidence: {self.stats['low_confidence_matches']}")
        logger.info(f"  No matches: {self.stats['no_matches']}")

        return matches_df

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get matching statistics.

        Returns:
            Statistics dictionary:
            {
                'total_matches': 150,
                'high_confidence_matches': 135,
                'low_confidence_matches': 10,
                'no_matches': 5,
                'match_rate': 0.97
            }
        """
        total_processed = (
            self.stats['total_matches'] + self.stats['no_matches']
        )

        match_rate = (
            self.stats['total_matches'] / total_processed
            if total_processed > 0 else 0
        )

        return {
            **self.stats,
            'match_rate': round(match_rate, 3)
        }

    def get_match_quality_report(self, matches_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate quality report for matches.

        Args:
            matches_df: DataFrame with match results

        Returns:
            Quality report with score distributions
        """
        if matches_df.empty:
            return {}

        report = {
            'total_matches': len(matches_df),
            'average_score': round(matches_df['match_score'].mean(), 2),
            'median_score': round(matches_df['match_score'].median(), 2),
            'min_score': round(matches_df['match_score'].min(), 2),
            'max_score': round(matches_df['match_score'].max(), 2),
            'confidence_distribution': matches_df['confidence'].value_counts().to_dict()
        }

        # Score ranges
        report['score_ranges'] = {
            '90-100': len(matches_df[matches_df['match_score'] >= 90]),
            '80-89': len(matches_df[(matches_df['match_score'] >= 80) & (matches_df['match_score'] < 90)]),
            '70-79': len(matches_df[(matches_df['match_score'] >= 70) & (matches_df['match_score'] < 80)]),
            '60-69': len(matches_df[(matches_df['match_score'] >= 60) & (matches_df['match_score'] < 70)]),
            'below_60': len(matches_df[matches_df['match_score'] < 60])
        }

        return report


# Usage example
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Example configuration
    config = {
        'fuzzy_threshold': 65,
        'weights': {
            'ratio': 0.4,
            'token_set': 0.5,
            'partial': 0.1
        },
        'high_confidence': 90,
        'manual_review_threshold': 70
    }

    # Initialize matcher
    matcher = FuzzyMatcher(config)

    # Test normalization
    print("\n=== Name Normalization ===")
    print(f"Original: 'Крепость Нарикала!'")
    print(f"Normalized: '{matcher.normalize_name('Крепость Нарикала!')}'")

    # Test single match
    print("\n=== Single Match ===")
    score = matcher.match("Нарикала", "Narikala")
    print(f"'Нарикала' <-> 'Narikala': {score}")

    score = matcher.match("Светицховели", "Svetitskhoveli")
    print(f"'Светицховели' <-> 'Svetitskhoveli': {score}")

    # Test best match finding
    print("\n=== Best Match Finding ===")
    targets = ["Narikala", "Vardzia", "Ananuri", "Uplistsikhe"]

    match, score = matcher.find_best_match("Нарикала", targets)
    print(f"Best match for 'Нарикала': {match} (score: {score})")

    match, score = matcher.find_best_match("Вардзия", targets)
    print(f"Best match for 'Вардзия': {match} (score: {score})")

    # Test dataframe matching
    print("\n=== DataFrame Matching ===")
    df_ru = pd.DataFrame({
        'name_ru': ['Нарикала', 'Вардзия', 'Ананури', 'Уплисцихе'],
        'location_ru': ['Тбилиси', 'Самцхе-Джавахети', 'Мцхета-Мтианети', 'Шида-Картли']
    })

    df_en = pd.DataFrame({
        'name_en': ['Narikala Fortress', 'Vardzia Cave Monastery', 'Ananuri Castle', 'Uplistsikhe Cave Town'],
        'location_en': ['Tbilisi', 'Samtskhe-Javakheti', 'Mtskheta-Mtianeti', 'Shida Kartli']
    })

    matches_df = matcher.match_dataframes(
        df_ru, df_en,
        source_column='name_ru',
        target_column='name_en',
        source_name='ru',
        target_name='en'
    )

    print("\nMatch Results:")
    print(matches_df[['name_ru', 'name_en', 'match_score', 'confidence']])

    # Statistics
    print("\n=== Statistics ===")
    print(matcher.get_statistics())

    # Quality report
    print("\n=== Quality Report ===")
    report = matcher.get_match_quality_report(matches_df)
    for key, value in report.items():
        print(f"{key}: {value}")