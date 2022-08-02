import sys

from loguru import logger


def log_init(cfg, output_folder):
    handlers = []
    if cfg.PARALLEL.IS_MASTER:
        handlers = [
            #{"sink": sys.stdout, "format": "{time:[MM-DD HH:mm:ss]} - {message}"},
            dict(sink=sys.stdout, format="<green>{time:MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",),
            dict(sink=f'{output_folder}/logs.txt', format="<green>{time:MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",),
            #{"sink": f"{output_folder}/logs.txt", "format": "{time:[MM-DD HH:mm:ss]} - {message}"},
        ]
    logger.configure(**{"handlers": handlers})
