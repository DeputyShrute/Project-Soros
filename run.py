 #!/usr/bin/env python -W ignore::DeprecationWarning
from scripts.TUI import menu
import os

def __init__():
    os.remove("/home/ryan/Documents/Python/Project-Soros/darknet/data/test.txt & /home/ryan/Documents/Python/Project-Soros/darknet/data/result.txt")

if __name__ == "__main__":
    os.system('cls')
    os.system('clear')
    menu.main_menu()