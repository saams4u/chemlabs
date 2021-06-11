import os
import logging
import logging.config
import utils
import time

# Directories
BASE_DIR = os.path.normpath(os.getcwd() + os.sep + os.pardir)
APP_DIR = os.getcwd()  # project root
LOGS_DIR = os.path.join(APP_DIR, 'logs')
EXPERIMENTS_DIR = os.path.join(APP_DIR, 'experiments')

# Create dirs
utils.create_dirs(LOGS_DIR)
utils.create_dirs(EXPERIMENTS_DIR)

# Loggers
log_config = utils.load_json(filepath=os.path.join(BASE_DIR, 'logging.json'))
logging.config.dictConfig(log_config)
logger = logging.getLogger('logger')
