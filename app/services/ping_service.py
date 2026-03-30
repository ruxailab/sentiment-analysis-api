"""
This is a simple service that returns a message when pinged.
"""

class PingService:
    def __init__(self) -> None:
        pass

    def ping(self) -> dict[str, str]:
        """
        Returns a ping response.

        Returns:
            dict[str, str]: A dictionary containing a 'message' key with 'Pong!' as value.
        """
        return {
            "message": "Pong!"
        }
    
# if __name__ == "__main__":
#     ping_service = PingService()

#     result = ping_service.ping()
#     print("result",result)

#  Run:
#  python -m app.services.ping_service