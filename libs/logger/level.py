"""Read the log level from the LOG_LEVEL environment variable, WARNING by default.

Used by libs/control/learnt_controller.py to set its logger level.
"""

import os

# check if the environment variable is set
if 'LOG_LEVEL' in os.environ:
    LOG_LEVEL = os.environ['LOG_LEVEL']
else:
    LOG_LEVEL = "WARNING"
