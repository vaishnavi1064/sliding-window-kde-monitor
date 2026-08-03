class Bucket:
    __slots__ = ("size", "timestamp")

    def __init__(self, timestamp: int):
        self.size = 1
        self.timestamp = timestamp
