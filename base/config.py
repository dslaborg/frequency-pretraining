class Config:
    """Singleton class for the configuration."""

    __instance = None

    def __init__(self):
        """Virtually private constructor."""
        raise Exception("This class is a singleton!")

    @staticmethod
    def get():
        if Config.__instance is None:
            raise ValueError("Config was not initialized")
        return Config.__instance

    @staticmethod
    def initialize(config):
        Config.__instance = config
