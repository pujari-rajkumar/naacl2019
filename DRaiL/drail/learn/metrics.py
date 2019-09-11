from prettytable import PrettyTable

class Metrics():

    def __init__(self):
        self.metrics = {}

    def add_classifier(self, name):
        self.metrics[name] = {'gold_data': [], 'pred_data': []}

    def reset_metrics(self):
        for name in self.metrics.keys():
            self.metrics[name] = {'gold_data': [], 'pred_data': []}

    def load_metrics(self, local_metrics):
        for name in local_metrics:
            self.load_metrics_classif(name, local_metrics[name])

    def load_metrics_classif(self, name, local_metrics):
        if name not in self.metrics:
            self.add_classifier(name)
        self.metrics[name]['gold_data'] += local_metrics['gold_data']
        self.metrics[name]['pred_data'] += local_metrics['pred_data']

