from loguru import logger
import os
import glob

def limit_log_files(directory="./logs", pattern="logs_*.log", max_files=10):
    files = sorted(glob.glob(os.path.join(directory, pattern)), key=os.path.getmtime)
    if len(files) > max_files:
        for f in files[:-max_files]:
            os.remove(f)

logger.add(
    "./logs/logs_{time}.log",
    rotation="5 MB",
    retention=lambda _: limit_log_files(max_files=10),
    level="INFO"
)
