import json
from io import StringIO
import random
import webbrowser
import tempfile
import os

_HERE            = os.path.dirname(os.path.abspath(__file__))
_TEST_FILEPATH   = os.path.join(_HERE, 'examples', 'test.json')
_CSS_DEFAULTPATH = os.path.join(_HERE, 'style.css')


class HTML_Writer:

    def __init__(self, cssfile = _CSS_DEFAULTPATH):
        self.io = StringIO()
        self.make_pretty = False
        self._css_path = cssfile
            
    @property
    def css(self):
        with open(self._css_path) as f:
            return f.read()

    def println(self, x):
        self.io.write(x)
        self.io.write('\n')

    def print(self, x):
        self.io.write(x)

    def render_scalar(self, x, cls="string"):
        self.print('<span class={}>{}</span>'.format(cls, x))

    def render_num(self, x):
        self.render_scalar(x, 'num')

    def render_string(self, x):
        x = '"' + x + '"'
        x = html_escape(x)
        self.render_scalar(x, 'string')
    
    def render_key(self, x):
        x = '"' + x + '"' + ': '
        x = html_escape(x)
        self.render_scalar(x, 'key')

    def render_null(self, x):
        self.render_scalar('null', 'null')

    def render_bool(self, x):
        self.render_scalar(x, 'bool')

    def render_list(self, x):
        if len(x) == 1:
            self.render_item(x[0])
            return
        if len(x) == 0:
            return

        self.println('<ul>')
        for item in x:
            self.print('<li>')
            self.render_item(item)
            self.println('</li>')

        self.println('</ul>')

    def render_toggle(self):
        s = rand_string()
        s = """
        <input type="checkbox" unchecked id="{}"/>
        <label for="{}"></label>
        """.format(s,s)

        self.println(s)
        
    def render_object(self,x):
        if len(x) == 0:
            return

        self.println('<ul>')
        for key,val in x.items():
            self.print('<li>')
            if is_collapsible(val):
                self.render_toggle()
            self.render_key(str(key))
            self.render_item(val)
            self.println('</li>')
        self.println('</ul>')

    def render_item(self,x):
        x = convert(x)

        if type(x) is dict:
            self.render_object(x)
        elif islistlike(x):
            self.render_list(x)
        elif type(x) is bool:
            self.render_bool(x)
        elif x is None:
            self.render_null(x)
        elif type(x) is float:
            self.render_num(x)
        else:
            self.render_string(x)
    
    def get_html(self, d):
        html_code = \
        """
        <html>
            <head>
            <link href="https://fonts.googleapis.com/css?family=Roboto+Mono" rel="stylesheet">
            <style>
                {}
            </style>
            </head>
            <body>
                    {}
            </body>
        </html
        """

        self.io.seek(0)
        self.io.truncate(0)
        self.render_object(d)
        self.io.seek(0)
        s = self.io.read()

        s = html_code.format(self.css, s)

        return s

    def write_html(self, d, filename='test.html'):
        html_code = self.get_html(d)
        with open(filename,'w+') as f:
            f.write(html_code)

    def view_html(self, d, filename = None):
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.html') as f:
            f.write(self.get_html(d))

        webbrowser.open_new(f.name)

        
def is_collapsible(val):
    if (isdict(val)):
        return True
    if (islistlike(val)) and (len(val) > 1):
        return True

    return False

def rand_string():
    return ''.join(random.choice('abcdedghijklmnopqrstuvwxyz') for i in range(20))


def convert(x):
    if islistlike(x):
        return x
    if isdict(x):
        return dict(x)

    if type(x) is tuple:
        return list(x)
    
    if x is None:
        return None

    try:
        return float(x)
    except:
        pass

    if x.upper() == 'TRUE':
        return True
    
    if x.upper() == 'FALSE':
        return False
    
    if x.upper() == 'NONE' or x.upper() == 'NULL':
        return None
    
    return str(x)

class convert_json:

    def __init__(self, fun):
        self.fun = fun

    def __call__(self, d, *args, **kwargs):
        if type(d) is not dict: # assume filepath
            with open(d) as f:
                d = json.load(f)
            
        self.fun(d, *args, **kwargs)

html_escape_table = {
       "&": "&amp;",
       '"': "&quot;",
       "'": "&apos;",
       ">": "&gt;",
       "<": "&lt;",
       }
   
def html_escape(text):
    """Produce entities within text."""
    return "".join(html_escape_table.get(c,c) for c in text)

def clean_dict(d):
    """
    Clean up dictionary tree by removing empty entries and merging nodes
    that only have one child, e.g.

    x
    / \
    y   z
        \
        a

    becomes
    x
    / \
    y   a

    They keys get merged like key1.key2 for clarity

    """
    if islistlike(d):
        dnew = []
        for i in range(len(d)):
            dnew.append(clean_dict(d[i]))

        return dnew

    if isdict(d):
        dnew = {}
        
        if (len(d) == 1) and isdict(next(iter(d.values()))):
            parent_key = next(iter(d.keys()))
            sub_dict   = next(iter(d.values()))
            d.popitem()
            for key,val in sub_dict.items():
                new_key = '.'.join((parent_key, key))
                d[new_key] = val

        for key, val in d.items():
            if (val != False) and (bool(val) == False):
                continue
            else:
                dnew[key] = clean_dict(d[key])
    
        return dnew

    return d
    
def isdict(d):

    if issubclass(type(d),dict):
        return True
    
    return False

def islistlike(x):
    if issubclass(type(x), list) or issubclass(type(x),tuple):
        return True

    else:
        return False

def ensure_all_keys_are_strings(d):

    if isdict(d):
        d_new = {}
        for key,val in d.items():
            d_new[str(key)] = ensure_all_keys_are_strings(d[key])
        return d_new
    elif islistlike(d):
        d_new = []
        for val in d:
            d_new.append(ensure_all_keys_are_strings(val))
        return d_new
    else:
        return d


@convert_json
def view(d, cssfile = _CSS_DEFAULTPATH, clean = True):
    """
    This function takes either a dictionary or path to .json file as input and
    opens up an interactive tree-like browser in your web browser.

    Example
    -------
    Display a dict in web browser:

        view(some_dict)

    Or pass a path to a .JSON file

        view('some_file.json') 

    """
    d = ensure_all_keys_are_strings(d)
    if clean:
        d = clean_dict(d)

    h = HTML_Writer(cssfile=cssfile)
    h.view_html(d)

@convert_json
def write(d, filename, cssfile = _CSS_DEFAULTPATH, clean = True):
    """
    This function takes either a dictionary or path to .json file as input and
    saves an interactive HTML file.

    Example
    -------
    Pass a python dictionary object

        write(some_dict)

    Or pass a path to a .JSON file

        write('some_file.json') 

    """
    d = ensure_all_keys_are_strings(d)
    if clean:
        d = clean_dict(d)

    h = HTML_Writer(cssfile = cssfile)
    h.write_html(d, filename)

if __name__ == '__main__':

    with open(_TEST_FILEPATH) as f:
        fulltext = f.read()
        d = json.loads(fulltext)

    h = HTML_Writer()
    h.view_html(d)

        
    