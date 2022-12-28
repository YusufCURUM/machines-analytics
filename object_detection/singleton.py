class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        key = (cls, args, str(kwargs))  # key is a tuple of the class, arguments, and keyword arguments
        # key = args[0] if args else None
        if key not in cls._instances:
            cls._instances[key] = super().__call__(*args, **kwargs)
        return cls._instances[key]
