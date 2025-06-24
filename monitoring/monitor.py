import pandas as pd
import time

class RealTimeMonitor:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path, parse_dates=['RealtimeClockDateandTime'])
        self.index = 0

    def get_next_reading(self):
        if self.index >= len(self.df):
            return None  # End of stream
        row = self.df.iloc[self.index]
        self.index += 1
        return row.to_dict()

    def get_history(self, limit=100):
        return self.df.iloc[max(0, self.index - limit):self.index]

    def reset(self):
        self.index = 0
