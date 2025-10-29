"""
Run Enrichment Script

Advanced command-line script for running enrichment pipeline with custom options.

Usage:
    # Run with default config
    python scripts/run_enrichment.py

    # Run with custom source
    python scripts/run_enrichment.py --source AIAnastasia/Georgian-attractions

    # Run specific steps only
    python scripts/run_enrichment.py --steps ner category

    # Run with custom output
    python scripts/run_enrichment.py --output ./my_results/enriched.csv

    # Run with custom config
    python scripts/run_enrichment.py --config ./my_config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipelines.enrichment_pipeline import EnrichmentPipeline


def setup_logging(verbose: bool = False):
    """
    Setup logging configuration.

    Args:
        verbose: Enable verbose (DEBUG) logging
    """
    log_dir = Path('./logs')
    log_dir.mkdir(exist_ok=True)

    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'enrichment.log'),
            logging.StreamHandler()
        ]
    )


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Run Georgian Attractions enrichment pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python scripts/run_enrichment.py
  
  # Run with HuggingFace dataset
  python scripts/run_enrichment.py --source AIAnastasia/Georgian-attractions
  
  # Run with local CSV file
  python scripts/run_enrichment.py --source ./data/attractions.csv
  
  # Run only specific enrichment steps
  python scripts/run_enrichment.py --steps ner category
  
  # Run with custom output path
  python scripts/run_enrichment.py --output ./results/enriched_data.csv
  
  # Run with custom configuration
  python scripts/run_enrichment.py --config ./custom_config.yaml
  
  # Enable verbose logging
  python scripts/run_enrichment.py --verbose
        """
    )

    parser.add_argument(
        '-c', '--config',
        type=str,
        default='./config/config.yaml',
        help='Path to configuration YAML file (default: ./config/config.yaml)'
    )

    parser.add_argument(
        '-s', '--source',
        type=str,
        default=None,
        help='Data source (HuggingFace dataset, local file, or URL). Overrides config.'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output file path for enriched data. Overrides config.'
    )

    parser.add_argument(
        '--steps',
        nargs='+',
        choices=['ner', 'category', 'tags', 'fuzzy_match'],
        default=None,
        help='Specific enrichment steps to run (default: all)'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose (DEBUG) logging'
    )

    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save results (useful for testing)'
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()

    # Setup logging
    setup_logging(verbose=args.verbose)
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("GEORGIAN ATTRACTIONS ENRICHMENT PIPELINE")
    logger.info("=" * 80)

    try:
        # Check if config exists
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error(f"Configuration file not found: {config_path}")
            logger.info("Please create the configuration file before running.")
            logger.info("Example config can be found in: config/config.yaml")
            return 1

        # Log configuration
        logger.info(f"\nConfiguration:")
        logger.info(f"  Config file: {args.config}")
        logger.info(f"  Source: {args.source or 'from config'}")
        logger.info(f"  Output: {args.output or 'from config'}")
        logger.info(f"  Steps: {args.steps or 'all'}")
        logger.info(f"  Verbose: {args.verbose}")

        # Initialize pipeline
        logger.info(f"\nInitializing pipeline...")
        pipeline = EnrichmentPipeline(config_path=str(config_path))

        # Run pipeline
        logger.info("\nRunning enrichment pipeline...")

        enriched_df = pipeline.run(
            source=args.source,
            output_path=args.output,
            steps=args.steps
        )

        # Results
        logger.info("\n" + "=" * 80)
        logger.info(" PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"\nResults:")
        logger.info(f"  Total records: {len(enriched_df)}")
        logger.info(f"  Output location: ./data/processed/")

        if args.output:
            logger.info(f"  Custom output: {args.output}")

        # Print component statistics
        stats = pipeline.get_statistics()

        logger.info(f"\nComponent Statistics:")
        logger.info(f"  NER entities extracted: {stats['ner_processor']['with_entities']}")
        logger.info(f"  Categories classified: {stats['category_classifier']['processed']}")
        logger.info(f"  Tags generated: {stats['tag_generator']['processed']}")
        logger.info(f"  Names matched: {stats['fuzzy_matcher']['total_matches']}")

        logger.info(f"\nProcessing time: {stats['pipeline']['processing_time']}s")

        return 0

    except KeyboardInterrupt:
        logger.warning("\n\n  Pipeline interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"\n Pipeline failed with error:", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())