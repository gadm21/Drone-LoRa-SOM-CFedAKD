



import os 
import argparse
import json


conf_file = os.path.abspath("conf/EMNIST_balance_conf.json")


def update_config(conf_file, config_dict):
    with open(conf_file, "r") as f:
            old_conf_dict = json.loads(f.read())
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

    if len(args.args) % 2 != 0:
        raise ValueError("Expected key-value pairs but received an odd number of arguments.")

    config_dict = {}
    for i in range(0, len(args.args), 2):
        key = args.args[i]
        value = str_to_float_or_bool(args.args[i+1])
        config_dict[key] = value

    return config_dict

if __name__ == "__main__":
    config_dict  = main()
    update_config(conf_file, config_dict)
