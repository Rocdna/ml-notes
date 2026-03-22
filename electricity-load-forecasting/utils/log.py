from loguru import logger
import sys
from pathlib import Path

# 日志目录
LOG_DIR = Path(__file__).parent.parent / "log"
LOG_DIR.mkdir(exist_ok=True)


class Logger:
    """日志类，用法：logger = Logger('train') 或 Logger('predict')"""

    def __init__(self, name: str = 'train'):
        self.name = name
        self.log = logger
        self.log.remove()

        # 控制台输出
        self.log.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
            level="INFO"
        )

        # 文件输出
        self.log.add(
            LOG_DIR / f"{self.name}_{{time:YYYY-MM-DD}}.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            level="DEBUG",
            rotation="00:00",
            retention="30 days",
            encoding="utf-8"
        )

    def info(self, msg):
        self.log.info(msg)

    def debug(self, msg):
        self.log.debug(msg)

    def warning(self, msg):
        self.log.warning(msg)

    def error(self, msg):
        self.log.error(msg)


if __name__ == '__main__':
    print('hello log!')
