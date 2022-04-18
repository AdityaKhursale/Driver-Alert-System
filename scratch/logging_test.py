import yaml
import logging
import logging.config

with open("logging_config.yml", 'rt') as f:
    config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)

# create logger
logger = logging.getLogger('driver_alert')

# 'application' code
logger.debug('debug message')
logger.info('info message')
logger.warning('warn message')
logger.error('error message')
logger.critical('critical message')