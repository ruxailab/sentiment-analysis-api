"""
This is a simple service that returns a message when pinged.
"""

class PingService:
    def __init__(self):
        pass

    def ping(self):
        return {
            "message": "Pong!"
        }