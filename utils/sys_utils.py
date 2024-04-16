import json
import re

###### Python Things ######

class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


#Chat-GPT generated tool for nested lists
def extract_configs(d):
    configs = [{}]
    for key, value in d.items():
        if isinstance(value, dict):
            sub_configs = extract_configs(value)
            configs = [{**c, key: sub} for c in configs for sub in sub_configs]
        elif isinstance(value, list):
            configs = [{**c, key: val} for c in configs for val in value]
        else:
            configs = [{**c, key: value} for c in configs]
    return configs


def sanitize_dict_string(d: dict):
    json_string = json.dumps(d)
    sanitized_string = re.sub(r'([^a-zA-Z0-9]+)', '_', json_string)
    return sanitized_string


