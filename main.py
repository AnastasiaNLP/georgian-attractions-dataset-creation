"""
Main Entry Point
================

Simple script to run the Georgian Attractions enrichment pipeline.

Usage:
    python main.py
"""

import logging
from pathlib import Path
from src.pipelines.enrichment_pipeline import EnrichmentPipeline


def setup_logging():
    """Setup logging configuration."""
    log_dir = Path('./logs')
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'enrichment.log'),
            logging.StreamHandler()
        ]
    )


def main():
    """Main execution function."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("GEORGIAN ATTRACTIONS ENRICHMENT PIPELINE")
    logger.info("=" * 80)

    try:
        # Initialize pipeline with config file
        config_path = './config/config.yaml'

        if not Path(config_path).exists():
            logger.error(f"Configuration file not found: {config_path}")
            logger.info("Please create config/config.yaml before running the pipeline.")
            return

        logger.info(f"Loading configuration from: {config_path}")
        pipeline = EnrichmentPipeline(config_path=config_path)

        # Run pipeline
        logger.info("\nStarting enrichment process...")
        enriched_df = pipeline.run()

        # Print results
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"\n✓ Processed {len(enriched_df)} records")
        logger.info(f"✓ Results saved to: ./data/processed/")

        # Print statistics
        stats = pipeline.get_statistics()
        logger.info("\nPipeline Statistics:")
        logger.info(f"  Processing time: {stats['pipeline']['processing_time']}s")
        logger.info(f"  Success rate: {stats['data_loader']['validation_rate']}")

    except Exception as e:
        logger.error(f"\n Pipeline failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()