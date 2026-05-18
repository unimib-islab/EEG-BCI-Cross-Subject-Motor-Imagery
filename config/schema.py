class Config:
    """Dot-access wrapper for nested dictionaries from YAML config files."""

    def __init__(self, d: dict):
        for k, v in d.items():
            if isinstance(v, dict):
                if 'range' in v:
                    start, stop = v['range']
                    step = v.get('step', 1)
                    exclude = set(v.get('exclude', []))
                    v = [x for x in range(start, stop, step) if x not in exclude]
                else:
                    v = Config(v)
            elif isinstance(v, str) and v.startswith("range("):
                v = list(eval(v, {"__builtins__": {}}, {"range": range}))
            setattr(self, k, v)

    def to_dict(self) -> dict:
        out = {}
        for k, v in self.__dict__.items():
            out[k] = v.to_dict() if isinstance(v, Config) else v
        return out

    def __repr__(self):
        return f"Config({self.__dict__})"