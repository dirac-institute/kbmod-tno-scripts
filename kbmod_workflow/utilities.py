def configure_logger(name, file_path):
    '''
    Simple function that will create a logger object and configure it to write
    to a file at the specified path. 
    Note: We import logging within the function because we expect this to be
    called within a parsl app.'''

    import logging

    logger = logging.getLogger(name)
    handler = logging.FileHandler(file_path)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    #? The following will cause _some_ parsl logging to end up in the same file.
    # logging.basicConfig(filename=file_path,
    #                     level=logging.DEBUG,
    #                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # logger = logging.getLogger(name)

    return logger
