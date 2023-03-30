import os
from texttable import Texttable

def load_configuration():
    """
    @returns: dictionary with the configuration of audio's countries source and chunks's source
    """
    with open('../conf.txt', 'r') as file:
        conf = file.read()

    conf_dict = {}
    conf = conf.split(':')
    for line in conf[1:]:
        line = line.split('->')
        key = line[0].replace(' ','')
        value = line[1][:len(line[1]) - 1].replace(' ','')
        conf_dict[key] = value 

    return conf_dict

def save_configuration(conf_dict):
    """
    @conf_dict: dictionary with the configuration of audio's countries source and chunks's source
    """
    with open('../conf.txt', 'w') as file:
        for key in conf_dict:
            line = ':{} -> {}\n'.format(key, conf_dict[key])
            file.write(line)

def show_info(header):
    table = Texttable()
    table.header(header)
    table.add_row(['Press X to exit'])

    in_value = -1
    while (in_value != 'x' and in_value != 'X'):
        os.system('cls' if os.name == 'nt' else 'clear')
        print(table.draw())
        in_value = input('\nOPTION:')