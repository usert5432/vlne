import json

class ConfigBase:

    __slots__ = []

    def to_dict(self):
        return { x : getattr(self, x) for x in self.__slots__ }

    def to_json(self, **kwargs):
        return json.dumps(self, default = lambda x : x.to_dict(), **kwargs)

    def __str__(self):
        return self.to_json(sort_keys = True)

    def pprint(self):
        return self.to_json(sort_keys = True, indent = 4)

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, item, value):
        return setattr(self, item, value)

