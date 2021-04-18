 #!/usr/bin/env python -W ignore::DeprecationWarning
from scripts.TUI import menu
import os

if __name__ == "__main__":
    os.system('cls')
    os.system('clear')
    menu.main_menu()