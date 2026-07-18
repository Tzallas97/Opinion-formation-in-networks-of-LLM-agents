from tkinter import _Any
def __getattr__(name):
    def f(*a, **k): return _Any()
    return f
class ScrolledText(_Any): pass
class Dialog(_Any): pass
