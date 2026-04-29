import logging
logger = logging.getLogger(__name__)

from indexer import run_indexer
from evaluation import evaluation

if __name__ == "__main__":
    logger.info(f"Prozess startet: Indexing")
    run_indexer()
    logger.info(f"Prozess startet: Evaluation")
    evaluation()