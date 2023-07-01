from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import yaml


# subclass dict and define getter-setter. This behaves as both dict and obj
class AttrDict(dict):
    def __init__(self, d):
        super(AttrDict, self).__init__(d)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = AttrDict(value)

    def __getattr__(self, key):
        if key in self:
            return self[key]
        else:
            #return self.__dict__[key]
            raise AttributeError(key)

    def __setattr__(self, key, value):
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self[key] = value

    def __delattr__(self, key):
        if key in self:
            del self[key]


def parse_yaml(file):
    with open(file, 'r') as fopen:
        yaml_config = AttrDict(yaml.load(fopen))
    return yaml_config
