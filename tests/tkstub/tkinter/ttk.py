from tkinter import _Any
class Frame(_Any): pass
class Label(_Any): pass
class Entry(_Any): pass
class Button(_Any): pass
class Combobox(_Any): pass
class Checkbutton(_Any): pass
class Radiobutton(_Any): pass
class Notebook(_Any): pass
class Scrollbar(_Any): pass
class Style(_Any): pass
class Treeview(_Any): pass
class Progressbar(_Any): pass
class Separator(_Any): pass
class LabelFrame(_Any): pass
class Labelframe(_Any): pass
class PanedWindow(_Any): pass
class Spinbox(_Any): pass
class Scale(_Any): pass
def __getattr__(name):
    def f(*a, **k): return _Any()
    return f
