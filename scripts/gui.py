import tkinter
from data_scraper import get_data
from model_start import Model


def call():
        a = startLabelEntry.get()
        b = starttimeEntry.get()
        b = int(b)
        c = startcolumnEntry.get()
        d = startmodelEntry.get()

        Open = Model(a,b,c,d)
        Open.data()


root=tkinter.Tk()
root.geometry('400x400')
startLabel =tkinter.Label(root,text="Enter Pair: ")
startLabelEntry=tkinter.Entry(root)
starttime =tkinter.Label(root,text="Enter Timestep: ")
starttimeEntry=tkinter.Entry(root)
startcolumn =tkinter.Label(root,text="Enter Column: ")
startcolumnEntry=tkinter.Entry(root)
startmodel =tkinter.Label(root,text="Enter Model: ")
startmodelEntry=tkinter.Entry(root)


startLabel.pack()
startLabelEntry.pack()
starttime.pack()
starttimeEntry.pack()
startcolumn.pack()
startcolumnEntry.pack()
startmodel.pack()
startmodelEntry.pack()

plotButton= tkinter.Button(root,text="RUN", command=call)

plotButton.pack()

root.mainloop()