from tkinter import *
from tkinter import ttk 


root = Tk()
frm = ttk.Frame(root, padding = 10)
frm.grid()
ttk.Label(frm, text = "Data Processing GUI").grid(column = 0, row = 0)
ttk.Button(frm, text = "Quit", command = root.destroy).grid(column = 10, row = 100)
root.mainloop()
