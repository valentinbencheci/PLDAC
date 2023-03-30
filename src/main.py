import os
import utils
import download_dataset
import file_manager
from texttable import Texttable

in_value = -1
table = Texttable()
table.header(['Menu'])
menu_items = ['1. Download birds audio', '2. Preprocess audio', '3. Show trained models', '4. Start model test', '5. Exit']
chunks_paths = ['../data/birds/signal_chunks/', '../data/other/signal_chunks/']

for item in menu_items:
    table.add_row([item])

while (in_value != 5):
    os.system('cls' if os.name == 'nt' else 'clear')
    print(table.draw())
    in_value = int(input('\nOPTION:'))

    if (in_value == 1):
        conf = utils.load_configuration()
        if (conf['countries'] == ''):
            download_dataset.download_save_all_meta_for_country()
            utils.show_info(['DATA WAS SUCCESSFULLY DOWNLOADED'])
            conf['countries'] = 'France, Germany, Italy'
            utils.save_configuration(conf)
        else:
            utils.show_info(['DATA IS ALREADY DOWNLOADED'])
    elif (in_value == 2):
        conf = utils.load_configuration()
        if (conf['chunks'] == ''):
            chunks_paths = ['../data/birds/signal_chunks/', '../data/other/signal_chunks/']
            for path in chunks_paths:
                if (not os.path.exists(path)):
                    os.makedirs(path)
            file_manager.preprocess_all_original_audio()
            conf['chunks'] = 'true'
            utils.save_configuration(conf)
        else:
            utils.show_info(['AUDIO IS ALREADY PREPROCESSED'])
    elif (in_value == 3):
        pass
    elif(in_value == 4):
        pass