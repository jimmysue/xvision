import logging


def get_logger(fullpath=None):
    logger = logging.getLogger(None)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s | %(message)s')

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if fullpath:
        fh = logging.FileHandler(fullpath, mode='a')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

set_logger = get_logger


if __name__ == '__main__':

    logger = get_logger('./log.txt')

    logger.info('this is first log')
    logger.debug('this is debug log')
    logger.error('this is error logstr')
