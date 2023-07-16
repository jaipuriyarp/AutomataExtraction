import time

class RecordTime():
    def __init__(self, record_elapsed_time=False):
        self.real_start = None
        self.start_time = None
        self.end_time = None
        self.end_time_list = []
        self.record_elapsed_time = record_elapsed_time
        self.elapsed_time_list = []
        if not(self.record_elapsed_time):
            print(f"WARNING: record_elapsed_time parameter is not enabled!!")
    def start(self):
        if self.real_start is None:
            self.real_start = time.time()
        self.start_time = time.time()
    def stop(self):
        self.end_time = time.time()
        elapsed_time = self.end_time - self.start_time
        print(f"Execution time: {elapsed_time:.4f} seconds")
        if self.record_elapsed_time:
            self.end_time_list.append(self.end_time)
            self.elapsed_time_list.append(elapsed_time)

    def reset(self):
        self.start_time = None
        self.end_time = None

    def report(self):
        if self.record_elapsed_time:
            self.end_time_list = [i - self.real_start for i in self.end_time_list]
            print(f"Elapsed time list: {self.elapsed_time_list}")
            print(f"End time list: {self.end_time_list}")
        else:
            raise Exception(f"Parameter 'record_elapsed_time' was not enabled, so no report "
                            f"on timing!")