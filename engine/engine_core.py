# engine_core.py
# Core game initialization and setup.

import logging
import sys
from datetime import datetime
from pathlib import Path
from engine.game_loop import GameLoop
from core.game_events import EventType
from core.game_events_manager import GameEventManager
from engine.game_state import GameState

def setup_logging():
    """
    Configure logging for the game.
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"game_{timestamp}.log"
    
    # Configure logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Configure file handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(log_format, date_format))
    
    # Configure console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    console_handler.setStream(sys.stdout)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Set encoding for stdout
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    
    return log_file

def main():
    """
    Main entry point for the game.
    """
    # Setup logging
    log_file = setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting AI Word Strategy Game")
    logger.info(f"Log file: {log_file}")
    
    try:
        # Initialize and start game
        game_loop = GameLoop()
        game_loop.start()
    except Exception as e:
        logger.error(f"Game crashed: {str(e)}", exc_info=True)
        raise
    finally:
        logger.info("Game ended")

if __name__ == "__main__":
    main() 