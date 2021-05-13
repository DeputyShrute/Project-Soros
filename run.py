 #!/usr/bin/env python -W ignore::DeprecationWarning
from scripts.TUI import menu
import os

def __init__():
    directo = os.path.dirname(__file__)
    path1 = os.path.join(directo, 'darknet/data/test.txt')
    path2 = os.path.join(directo, 'darknet/data/result.txt')
    os.remove(path1)
    os.remove(path2)

if __name__ == "__main__":
    os.system('cls')
    os.system('clear')
    menu.main_menu()