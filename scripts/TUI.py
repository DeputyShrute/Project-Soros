#!/usr/bin/env python -W ignore::DeprecationWarning
import os
from PIL import Image
import glob
from simple_term_menu import TerminalMenu
from scripts.create_data import make_chart
from scripts.data_scraper import get_data
from scripts.part_launch import launch


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
            elif select == 2:
                menu.select_pair()
            elif select == 3:
            # Price Prediction
                launch.predictions('EURUSD')
            # Exit
            elif select == 4:
                print("Exiting program")
                try:
                    dir = 'scripts/Finance_Data/Chart_Snapshot'
                    for pic in os.listdir(dir):
                        os.remove(os.path.join(dir, pic))
                    os.system('cls')
                    os.system('clear')
                    exit()
                finally:
                    os.system('cls')
                    os.system('clear')
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
                make_chart.load(select, currency_pairs)
            else:
                menu.main_menu()


if __name__ == "__main__":
    start = menu.main_menu()
