import logging

def configure_logger(name,log_file_name='ExSeq-Toolbox_logs.log'):

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)

    fhandler = logging.FileHandler(log_file_name,mode="a")
    fhandler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    fhandler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.addHandler(fhandler)

    return logger
