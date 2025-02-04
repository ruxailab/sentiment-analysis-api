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
    
# if __name__ == "__main__":
#     ping_service = PingService()

#     result = ping_service.ping()
#     print("result",result)

#  Run:
#  python -m app.services.ping_service