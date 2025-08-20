import asyncio
import logging
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from src.lunarcrush_ingestor import LunarCrushIngestor
from config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def run_ingestion():
    """Run ingestion for all configured symbols"""
    logger.info(f"Starting ingestion cycle at {datetime.now()}")
    
    ingestor = LunarCrushIngestor()
    
    # Determine which symbols to use
    symbols = settings.SYMBOLS if settings.SYMBOLS and settings.SYMBOLS[0] else settings.DEFAULT_SYMBOLS
    
    logger.info(f"Processing symbols: {symbols}")
    
    # Use the ingestor's built-in batch processing with rate limiting
    await ingestor.ingest_all_symbols(symbols)
    
    # Print stats after completion
    stats = ingestor.get_ingestion_stats()
    logger.info(f"Ingestion stats: {stats}")
    
    logger.info(f"Ingestion cycle completed at {datetime.now()}")

def run_sync_ingestion():
    """Wrapper to run async ingestion in sync context (for APScheduler)"""
    asyncio.run(run_ingestion())

async def main():
    logger.info("Starting LunarCrush ingestion scheduler")
    logger.info(f"Configuration: Host={settings.QDRANT_HOST}, Port={settings.QDRANT_PORT}")
    logger.info(f"Symbols: {settings.SYMBOLS if settings.SYMBOLS and settings.SYMBOLS[0] else settings.DEFAULT_SYMBOLS}")
    logger.info(f"Interval: {settings.INGESTION_INTERVAL} hour(s)")
    
    # Run once on startup
    await run_ingestion()
    
    # Create async scheduler
    scheduler = AsyncIOScheduler()
    
    # Schedule hourly runs
    scheduler.add_job(
        run_sync_ingestion,  # Use the sync wrapper
        'interval',
        hours=settings.INGESTION_INTERVAL,
        id='ingestion_job'
    )
    
    scheduler.start()
    
    try:
        logger.info(f"Scheduler started. Will run every {settings.INGESTION_INTERVAL} hour(s)")
        # Keep the program running
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")
        scheduler.shutdown()

if __name__ == "__main__":
    asyncio.run(main())