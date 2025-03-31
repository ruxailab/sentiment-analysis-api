
import logging
import os
import queue
from logging.handlers import QueueHandler, QueueListener

from app.config import Config

config = Config().config # Load the configuration

logs_dir = config.get('flask').get('logs_dir')

# Create a specific folder for logs
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

# Define log file paths
log_file = os.path.join(logs_dir, 'debug.log')
requests_file = os.path.join(logs_dir, 'requests.log')

# Create a log queue for async logging
log_queue = queue.Queue()

# Formatter for logs
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Function to sanitize log messages
def sanitize_message(message):
    try:
        return message.encode("ascii", "replace").decode("ascii")
    except Exception as e:
        return f"Message could not be sanitized: {e}"

# Custom logging handler to sanitize messages
class SanitizingQueueHandler(QueueHandler):
    def emit(self, record):
        record.msg = sanitize_message(str(record.msg))
        super().emit(record)

# --- Custom Filters ---
class ApplicationFilter(logging.Filter):
    def filter(self, record):
        # print(record.name)
        return record.name == "application_logger"

class RequestsFilter(logging.Filter):
    def filter(self, record):
        # print(record.name)
        return record.name == "requests_logger"

# --- Application Logger ---
# File handler for application logs
app_file_handler = logging.FileHandler(log_file)
app_file_handler.setFormatter(formatter)
app_file_handler.addFilter(ApplicationFilter())  # Add filter to route application logs

# Logger for application logic
logger = logging.getLogger("application_logger")
logger.setLevel(logging.DEBUG)
logger.addHandler(SanitizingQueueHandler(log_queue))

# --- Requests Logger ---
# File handler for request logs
requests_file_handler = logging.FileHandler(requests_file)
requests_file_handler.setFormatter(formatter)
requests_file_handler.addFilter(RequestsFilter())  # Add filter to route requests logs

# Logger for incoming and outgoing requests
logging.getLogger('requests').setLevel(logging.DEBUG)
logging.getLogger('urllib3').setLevel(logging.DEBUG)
requests_logger = logging.getLogger("requests_logger")
requests_logger.setLevel(logging.DEBUG)
requests_logger.addHandler(SanitizingQueueHandler(log_queue))


# Create and configure loggers for requests and urllib3 libraries
logging.getLogger('requests').setLevel(logging.DEBUG)
logging.getLogger('urllib3').setLevel(logging.DEBUG)

# Add handler to requests and urllib3 loggers
requests_handler = logging.FileHandler(requests_file)
requests_handler.setFormatter(formatter)
logging.getLogger('requests').addHandler(requests_handler)  # Add handler to requests logger

urllib3_handler = logging.FileHandler(requests_file)
urllib3_handler.setFormatter(formatter)
logging.getLogger('urllib3').addHandler(urllib3_handler)  # Add handler to urllib3 logger


# --- Queue Listener ---
# The QueueListener processes logs from the queue
queue_listener = QueueListener(log_queue, app_file_handler, requests_file_handler)
queue_listener.start()

