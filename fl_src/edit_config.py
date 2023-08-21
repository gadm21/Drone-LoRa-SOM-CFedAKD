



import os 
import argparse
import json


conf_file = os.path.abspath("conf/EMNIST_balance_conf.json")


def update_config(conf_file, config_dict):
    with open(conf_file, "r") as f:
            old_conf_dict = json.load(f)
    # Update the old config dictionary with the new values
    old_conf_dict.update(config_dict)
    # Write the updated config dictionary to the json file
    with open(conf_file, "w") as f:
        json.dump(old_conf_dict, f, indent=4)  # `indent=4` makes the output more readable by indenting it

    

    
def str_to_bool(s):
    if s.lower() == 'true':
        return True
    elif s.lower() == 'false':
        return False
    else:
        raise ValueError("Cannot convert {} to a boolean value".format(s))

def str_to_float_or_bool(s):
    try:
        # Try converting to float first
        if '.' in s : 
            return float(s)
        else:
            return int(s)
    except ValueError:
        # If not a float, convert to boolean
        return str_to_bool(s)

def main():
    parser = argparse.ArgumentParser(description="Parse key-value pairs into a dictionary.")
    
    parser.add_argument('args', nargs='+', help="Arguments in key value pairs")

    args = parser.parse_args()

    if len(args.args) % 2 == 0:
        raise ValueError("Expected file name + key-value pairs but received an even number of arguments.")

    conf_file = "conf/" + args.args[0] + ".json"
    config_dict = {}
    for i in range(1, len(args.args), 2):
        key = args.args[i]
        print(key, args.args[i+1])
        if args.args[i+1].lower() in ['true', 'false']:
            value = str_to_bool(args.args[i+1])
        elif '.' in args.args[i+1]: 
            value = float(args.args[i+1])
        elif args.args[i+1].isdigit():
            value = int(args.args[i+1])
        elif args.args[i+1].lower() in ['none', 'null']:
            value = None
        else : 
            try : 
                value = str(args.args[i+1])
            except :
                raise ValueError("Cannot convert {} to a boolean, float, or string value".format(args.args[i+1]))
        config_dict[key] = value

    return conf_file, config_dict

if __name__ == "__main__":
    conf_file, config_dict  = main()
    update_config(conf_file, config_dict)
