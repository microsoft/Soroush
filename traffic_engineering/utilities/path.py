from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import utilities


class Path(object):

    def __init__(self, path_list=None, amount=0, index=0):
        self.path = path_list
        if self.path is None:
            self.path = list()
        self.amount = amount
        self.index = index

    def add_node(self, node_name):
        self.path.append(node_name)

    def add_list_node(self, list_node):
        self.path.extend(list_node)

    def set_amount(self, amount):
        self.amount = amount
        self.index = 1

    def get_avg_amount(self, amount):
        self.amount = (self.amount * self.index + amount) * 1.0/(self.index + 1)
        self.index += 1

    def sum_amount_with(self, amount):
        self.amount += amount
        self.index += 1

    def get_last_node(self):
        return self.path[len(self.path) - 1]

    def copy(self, new_path):
        for node in self.path:
            new_path.add_node(node)
        new_path.amount = self.amount
        return new_path

    def get_path(self):
        return self.path

    def get_amount(self):
        return self.amount

    def reduce_amount(self, amount):
        self.amount -= amount

    def does_amount_remain(self):
        return self.amount != 0

    def is_identical_to(self, other_path):
        if len(self.get_path()) != len(other_path.get_path()):
            return False

        for i in range(len(self.get_path())):
            if self.get_path()[i] != other_path.get_path()[i]:
                return False
        return True

    def is_identical_to_node_list(self, node_list):
        for i in range(len(node_list)):
            if self.get_path()[i] != node_list[i]:
                return False
        return True

    def print(self):
        print(self.path)

    def get_string(self):
        return utilities.list_to_string(self.get_path())