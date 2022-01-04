import sys

from loguru import logger

LOGGER_FORMAT = '<green>{time:YY-MM-DD HH:mm:ss}</green> | ' \
                '<level>{level: <8}</level> | ' \
                '<level>{message}</level>'


def get_logger(save_log=False, log_name=None):
    logger.configure(handlers=[{'sink': sys.stderr, 'level': 'DEBUG', 'format': LOGGER_FORMAT}])
    if save_log:
        logger.add(log_name + '_{time:YYYY-MM-DD}.log', level='INFO', format='<level>{message}</level>', enqueue=True)
    return logger
