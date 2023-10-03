import logging


def set_logging_params(level: int):
    """
    Configure basic logging for the program
    """
    logging.basicConfig(format="[%(levelname)s] %(asctime)s: %(message)s".format(10), datefmt='%I:%M:%S %p',level=level)
