class Benchmark:
    
    def __init__(self, train_stream, test_stream, dataset_name, valid_stream=None):

        self.train_stream = train_stream
        self.test_stream = test_stream
        self.valid_stream = valid_stream
        self.dataset_name = dataset_name