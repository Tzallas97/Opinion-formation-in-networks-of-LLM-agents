from tkinter import _Any
class Font(_Any):
    def actual(self, *a, **k): return {"family": "stub", "size": 10}
    def configure(self, *a, **k): pass
    def cget(self, *a, **k): return 10
def nametofont(name, root=None): return Font()
def families(*a, **k): return []
