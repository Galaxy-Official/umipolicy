import logging
from datetime import datetime

import coloredlogs


def setup_logger(level: str = "INFO"):
    assert level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    log_level = getattr(logging, level)
    logger = logging.getLogger()
    logging.basicConfig(
        level=log_level,
        format="[%(levelname)s] [%(name)s] %(message)s",
        handlers=[
            logging.StreamHandler(),  # 输出到控制台
            logging.FileHandler(
                f"logs/decision_making_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
                mode="a",
                encoding="utf-8",
            ),  # 输出到文件
        ],
    )
    coloredlogs.DEFAULT_LOG_FORMAT = "[%(levelname)s] [%(name)s] %(message)s"
    coloredlogs.install(
        level=level,
        logger=logger,
    )
