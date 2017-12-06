class Struct(dict):
    """
    Convert a nested dict into the equivalent nested Python class.
    """

    def __init__(self, *args, **kwargs):
        super(Struct, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for key, value in arg.items():
                    self[key] = self._wrap(value)
        if kwargs:
            for key, value in kwargs.items():
                self[key] = self._wrap(value)

    def _wrap(self, value):
        if isinstance(value, (tuple, list, set, frozenset)):
            return type(value)(self._wrap(v) for v in value)
        elif isinstance(value, dict):
            return Struct(value)
        else:
            return value

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Struct, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Struct, self).__delitem__(key)
        del self.__dict__[key]


if __name__ == "__main__":
    a = {"foo": 4, "bar": {"bu": "one", "zo": True}}
    x = Struct(**a)
    print(type(x))
    print(type(x.bar))
    print(x.bar)
