import logging
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
from indexer import run_indexer
from evaluation import evaluation

load_dotenv()

if __name__ == "__main__":
    logger.info(f"Prozess startet: Indexing")
    run_indexer()
    logger.info(f"Prozess startet: Evaluation")
    evaluation()