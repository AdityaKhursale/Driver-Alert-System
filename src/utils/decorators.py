class Cached:
    def __init__(self, *args, **kwargs):
        self.cached_data = {}
    
    def __call__(self, func):
        # TODO: Cache values considering arguments
        def inner(*args, **kwargs):
            if func not in self.cached_data:
                res = func(*args, **kwargs)
                self.cached_data[func] = res
            return self.cached_data[func]
        return inner


def _create_type(meta, name, attrs):
    type_name = f'{name}Type'
    type_attrs = {}
    for k, v in attrs.items():
        if type(v) is _ClassPropertyDescriptor:
            type_attrs[k] = v
    return type(type_name, (meta,), type_attrs)


class ClassPropertyType(type):
    def __new__(meta, name, bases, attrs):
        Type = _create_type(meta, name, attrs)
        cls = super().__new__(meta, name, bases, attrs)
        cls.__class__ = Type
        return cls


class _ClassPropertyDescriptor(object):
    def __init__(self, fget, fset=None):
        self.fget = fget
        self.fset = fset

    def __get__(self, obj, owner):
        if self in obj.__dict__.values():
            return self.fget(obj)
        return self.fget(owner)

    def __set__(self, obj, value):
        if not self.fset:
            raise AttributeError("can't set attribute")
        return self.fset(obj, value)

    def setter(self, func):
        self.fset = func
        return self


def classproperty(func):
    return _ClassPropertyDescriptor(func)
