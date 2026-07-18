class _Any:
    def __init__(self, *a, **k): pass
    def __getattr__(self, name):
        if name in ("_last_child_ids", "children", "tk"): return {}
        # Private instance state must behave like real attribute lookup
        # (AttributeError until assigned), otherwise hasattr()-guarded lazy
        # init in the app under test is silently skipped.
        if name.startswith("_") or name.endswith(("_var", "_vars", "_combo", "_entry", "_btn", "_label", "_text", "_frame", "_listbox", "_canvas", "_tree")):
            raise AttributeError(name)
        def f(*a, **k): return _Any()
        return f
    def __call__(self, *a, **k): return _Any()
    def __iter__(self): return iter([])
    def __str__(self): return ""
    def __index__(self): return 0
    def __bool__(self): return False

class Variable:
    _default = ""
    def __init__(self, master=None, value=None, name=None):
        self._v = value if value is not None else self._default
    def get(self): return self._v
    def set(self, v): self._v = v
    def trace_add(self, *a, **k): pass
    def trace(self, *a, **k): pass

class StringVar(Variable): _default = ""
class IntVar(Variable): _default = 0
class DoubleVar(Variable): _default = 0.0
class BooleanVar(Variable): _default = False

class Tk(_Any):
    def __init__(self, *a, **k): pass
    def title(self, *a): pass
    def geometry(self, *a): pass
    def minsize(self, *a): pass
    def protocol(self, *a): pass
    def mainloop(self): pass
    def after(self, *a, **k): return "id"
    def bind(self, *a, **k): pass
    def option_add(self, *a): pass
    def columnconfigure(self, *a, **k): pass
    def rowconfigure(self, *a, **k): pass
    def winfo_screenwidth(self): return 1920
    def winfo_screenheight(self): return 1080

class Toplevel(_Any): pass
class Frame(_Any): pass
class Label(_Any): pass
class Entry(_Any): pass
class Button(_Any): pass
class Text(_Any): pass
class Canvas(_Any): pass
class Listbox(_Any): pass
class Menu(_Any): pass
class Checkbutton(_Any): pass
class Radiobutton(_Any): pass
class Spinbox(_Any): pass
class Scrollbar(_Any): pass
class PanedWindow(_Any): pass
class LabelFrame(_Any): pass
class Scale(_Any): pass

END = "end"; W = "w"; E = "e"; N = "n"; S = "s"; NSEW = "nsew"; EW = "ew"; NS = "ns"
LEFT = "left"; RIGHT = "right"; TOP = "top"; BOTTOM = "bottom"; BOTH = "both"; X = "x"; Y = "y"
NORMAL = "normal"; DISABLED = "disabled"; WORD = "word"; FLAT = "flat"; SUNKEN = "sunken"
HORIZONTAL = "horizontal"; VERTICAL = "vertical"; CENTER = "center"; ANCHOR = "anchor"; INSERT = "insert"; SEL = "sel"
TclError = Exception
