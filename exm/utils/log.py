import logging
from typing import Any

def configure_logger(name: str, log_file_name: str = 'ExSeq-Toolbox_logs.log') -> Any:
    """
    Configures and returns a logger with both stream and file handlers.

    This function sets up a logger to send log messages to the console and to a log file. The console will display
    messages with a level of INFO and higher, while the file will contain messages with a level of DEBUG and higher.

    :param name: Name of the logger to configure. Typically, this is the name of the module calling the logger.
    :type name: str
    :param log_file_name: Name of the log file where logs will be saved. Defaults to 'ExSeq-Toolbox_logs.log'.
    :type log_file_name: str
    :return: Configured logger object.
    :rtype: logging.Logger

    :raises OSError: If there is an issue with opening or writing to the log file.
    """
    try:
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        # Stream handler configuration
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)

        # File handler configuration
        fhandler = logging.FileHandler(log_file_name, mode="a")
        fhandler.setLevel(logging.DEBUG)

        # Formatter configuration
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        fhandler.setFormatter(formatter)
        
        # Add handlers to the logger if they haven't been added yet
        if not logger.handlers:
            logger.addHandler(handler)
            logger.addHandler(fhandler)

        return logger
    except OSError as e:
        raise OSError(f"Failed to configure logger with file {log_file_name}: {e}")
