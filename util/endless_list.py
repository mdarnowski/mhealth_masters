class EndlessList:
    def __init__(self, value):
        self.value = value

    def __getitem__(self, index):
        return self.value

    def __iter__(self):
        while True:
            yield self.value

    def copy(self):
        return self

    def pop(self, index=0):
        return self.value
