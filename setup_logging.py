def setup_logging():
    """Configure logging to output detailed information."""
    import logging
    log_format = '%(asctime)s [%(levelname)s] %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('protsolm_enhanced.log')
        ]
    )
    return logging.getLogger()
