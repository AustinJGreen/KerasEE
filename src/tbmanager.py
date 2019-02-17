import math

import tensorflow as tf


class TensorboardManager:

    def __init__(self, base_dir, number_of_batches):
        self.base_dir = base_dir
        self.number_of_batches = number_of_batches
        self.writer = tf.summary.FileWriter(logdir=base_dir)
        self.var_dict = {}

    def log_var(self, var_name, epoch, batch, value):
        if math.isnan(value):
            return

        if var_name not in self.var_dict:
            summary = tf.Summary()
            summary.value.add(tag=var_name, simple_value=None)
            self.var_dict[var_name] = summary

        summary = self.var_dict[var_name]
        summary.value[0].simple_value = value
        self.writer.add_summary(summary, (epoch * self.number_of_batches) + batch)
