 #!/usr/bin/env python -W ignore::DeprecationWarning
import os
from PIL import Image
import glob
from simple_term_menu import TerminalMenu
from scripts.create_data import make_chart
from scripts.data_scraper import get_data


class menu:

    def main_menu():
        main_menu_title = "Main Menu\n"
        menu_items = ["Update All Data",
                      "Full Analysis", "Candlestick Analysis", "Price Prediction", "Exit"]
        main_menu_cursor = ">"
        main_menu_exit = False

        main_menu = TerminalMenu(
            title=main_menu_title,
            menu_entries=menu_items,
            menu_cursor=main_menu_cursor
        )

        while not main_menu_exit:
            select = main_menu.show()

            # Update Data
            if select == 0:
                get_data.get_url()
            # Full Analysis
            elif select == 1:
                menu.select_pair()
            # Candlestick Analysis

            # Price Prediction

            # Exit
            elif select == 4:
                print("Exiting program")
                exit()

    def select_pair():
        select_pair_title = "Select Pair\n"

        os.chdir('scripts/Finance_Data/Raw_Data/')
        currency_pairs = glob.glob('*.csv')
        currency_pairs.sort()
        currency_pairs.append('Exit')

        select_pair_cursor = ">"
        select_pair_exit = False
        currency_menu = TerminalMenu(
            title=select_pair_title,
            menu_cursor=select_pair_cursor,
            menu_entries=currency_pairs)

        while not select_pair_exit:
            select = currency_menu.show()
            if select < (len(currency_pairs) - 1):
                X = make_chart.load(select, currency_pairs)
                os.chdir("/home/ryan/Documents/Python/Project-Soros/darknet")
                os.system("./darknet detector test data/obj.data cfg/yolo-obj.cfg backup/yolo-obj_final.weights -dont_show -ext_output < data/test.txt > data/result.txt")
                predictions = Image.open('predictions.jpg')
                predictions.show()
                os.wait()
                
            else:
                menu.main_menu()


if __name__ == "__main__":
    start = menu.main_menu()
