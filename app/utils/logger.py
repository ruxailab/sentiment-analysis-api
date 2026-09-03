import logging
import queue
from logging.handlers import QueueHandler, QueueListener

from app.config import Config

config = Config().config  # Load the configuration

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
        return record.name == "application_logger"

class RequestsFilter(logging.Filter):
    def filter(self, record):
        return record.name == "requests_logger"

# --- Application Logger ---
# Console handler for application logs
app_console_handler = logging.StreamHandler()
app_console_handler.setFormatter(formatter)
app_console_handler.addFilter(ApplicationFilter())

# Logger for application logic
logger = logging.getLogger("application_logger")
logger.setLevel(logging.DEBUG)
logger.addHandler(SanitizingQueueHandler(log_queue))

# --- Requests Logger ---
# Console handler for request logs
requests_console_handler = logging.StreamHandler()
requests_console_handler.setFormatter(formatter)
requests_console_handler.addFilter(RequestsFilter())

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
requests_handler = logging.StreamHandler()
requests_handler.setFormatter(formatter)
logging.getLogger('requests').addHandler(requests_handler)

urllib3_handler = logging.StreamHandler()
urllib3_handler.setFormatter(formatter)
logging.getLogger('urllib3').addHandler(urllib3_handler)


# --- Queue Listener ---
# The QueueListener processes logs from the queue
queue_listener = QueueListener(log_queue, app_console_handler, requests_console_handler)
queue_listener.start()
