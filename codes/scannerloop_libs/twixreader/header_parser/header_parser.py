import sys
from antlr4 import *
from .XProtLexer import XProtLexer
from .XProtParser import XProtParser
from .XProtParserVisitor import XProtParserVisitor
import numpy as np
import re
import yaml, json
import os
import sys
import itertools

def parse_string(s):
    
    input = InputStream(s)
    lexer = XProtLexer(input)
    stream = CommonTokenStream(lexer)
    parser = XProtParser(stream)
    tree = parser.header()
    
    xpv = XProtParserVisitor()
    d = xpv.visit(tree)

    return (d,tree)

def parse_file(f):
    
    s = f.read()
    f.close()

    return parse_string(s)

def mkdir_for_file(filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            pass

def dict2yaml(d, filename):
    mkdir_for_file(filename)
    with open(filename, 'w+') as f:
        yaml.dump(json.loads(json.dumps(d)), f, allow_unicode=True, default_flow_style=False)
        
def dict2json(d, filename):
    mkdir_for_file(filename)
    with open(filename, 'w+') as f:
        s = json.dumps(d, indent=4)
        string2file(s, filename)

def dict2txt(d, filename):
    keys = list(d.keys())
    vals = [d[key] for key in keys]
    keys_and_vals = itertools.chain(*zip(keys,vals))

    s = ('### {} ###\n\n{}'*len(keys)).format(*keys_and_vals)

    string2file(s, filename)


def string2file(s, filename):
    mkdir_for_file(filename)
    with open(filename, "w+") as fid:
        fid.write("{0}".format(s))
