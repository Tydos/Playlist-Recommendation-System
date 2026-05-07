import logging

def get_logger(name: str, log_file: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter(
            '%(asctime)s %(levelname)s [%(name)s] %(message)s', 
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # # File Handler 
        # log_dir = Path(__file__).parent.parent / "logs"
        # log_dir.mkdir(parents=True, exist_ok=True)
        # filename = f"{log_file}.log" if log_file else f"{name}.log"
        
        # file_handler = logging.FileHandler(log_dir / filename, mode='a')
        # file_handler.setFormatter(formatter)
        # logger.addHandler(file_handler)

    return logger