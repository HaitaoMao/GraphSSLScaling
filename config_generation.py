import os
import json
import numpy as np

def parse_search_space(space_file):
    search_space = {}
    if os.path.exists('./{}.json'.format(space_file)):
        with open('./{}.json'.format(space_file), 'r') as f:
            paras_dict = json.load(f)
            for name in paras_dict:
                paras_type = paras_dict[name]['type']
                if paras_type == 'choice':
                    # name type list
                    try:
                        search_space[name] = np.random.choice(paras_dict[name]['list'])
                    except:
                        raise TypeError('The space file does not meet the format requirements,\
                            when parsing choice type.')
                else:
                    raise TypeError('The space file does not meet the format requirements,\
                            when parsing an undefined type.')
    else:
        raise FileNotFoundError('The space file {}.json is not found. Please ensure \
            the config file is in the root dir and is a txt.'.format(space_file))
    return search_space

class JsonEncoder(json.JSONEncoder):
    """Convert numpy classes to JSON serializable objects."""

    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(JsonEncoder, self).default(obj)
        
def main():
    space_file = 'hyper_example'
    for i in range(20):
        search_sapce = parse_search_space(space_file)
        # print(object(search_sapce))
        config_file = "./random_config/config_{}.json".format(i)
        with open(config_file, "w") as w:
            w.write(json.dumps(search_sapce,ensure_ascii=False, cls=JsonEncoder))
            print("config_{} is generated".format(i))

if __name__=="__main__":
    main()
