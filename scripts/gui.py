import tkinter
from tkinter import messagebox
from typing import Text
import subprocess as sub
from data_scraper import get_data
from model_start import Models


def call():
        a = startLabelEntry.get()
        b = starttimeEntry.get()
        try:
                b = int(b)
        except:
                show_method  = getattr(messagebox, 'show{}'.format('warning'))
                show_method('Warning', 'Value needs to be an Integer')


        d = startmodelEntry.get()
        e = verboseLevelEntry.get()
        try:
                if e == 'On':
                        e = 2
                elif e == 'Off':
                        e = 0
                else:
                        e = 1
        except:
                show_method  = getattr(messagebox, 'show{}'.format('warning'))
                show_method('Warning', 'Value needs to be an Integer')

        Open = Models(a,b,d,e)
        Open.data()
## Drop Down Values
currency_pair = ["EURUSD", "EURJPY", "USDCHF", "GBPUSD", "USDJPY", "USDCAD", "AUDUSD", "EURGBP", "EURAUD", "EURCHF", "GBPCHF", "CADJPY", "GBPJPY",
                        "AUDNZD", "AUDCAD", "AUDCHF", "AUDJPY", "CHFJPY", "EURNZD", "EURCAD", "CADCHF", "NZDJPY", "NZDUSD", "GBPAUD", "GBPNZD", "NZDCAD", "NZDCHF"]
currency_pair = sorted(currency_pair)
columns = ['Open', 'Close', 'High', 'Low']

model = ['KNN', 'CNN', 'MLP', 'LSTM']

verbose_level = ['On', 'Off', 'Partial']

#Creates Window Instance
root=tkinter.Tk()
root.geometry('400x400')

# Adds values to the window
startLabel =tkinter.Label(root,text="Enter Pair: ")
startLabelEntry = tkinter.StringVar()
startLabelEntry.set('EURUSD')
startLabel.pack()
startLabelEntrydrop=tkinter.OptionMenu(root, startLabelEntry, *currency_pair)
startLabelEntrydrop.pack()

starttime =tkinter.Label(root,text="Enter Timestep: ")
starttimeEntry=tkinter.Entry(root)
starttime.pack()
starttimeEntry.pack()

startcolumn =tkinter.Label(root,text="Enter Column: ")
startcolumnEntry=tkinter.StringVar()
startcolumnEntry.set('Open')
startcolumn.pack()
startcolumnEntrydrop=tkinter.OptionMenu(root, startcolumnEntry, *columns)
startcolumnEntrydrop.pack()

startmodel =tkinter.Label(root,text="Enter Model: ")
startmodelEntry=tkinter.StringVar()
startmodelEntry.set('KNN')
startmodel.pack()
startmodelEntrydrop=tkinter.OptionMenu(root, startmodelEntry, *model)
startmodelEntrydrop.pack()

verboseLevel =tkinter.Label(root,text="Enter Verbose: ")
verboseLevelEntry=tkinter.StringVar()
verboseLevelEntry.set('On')
verboseLevel.pack()
verboseLevelEntrydrop=tkinter.OptionMenu(root, verboseLevelEntry, *verbose_level)
verboseLevelEntrydrop.pack()

plotButton= tkinter.Button(root,text="RUN", command=call)
plotButton.pack()

root.mainloop()