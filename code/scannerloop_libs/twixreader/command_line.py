from twixreader.misc import twix2html
import sys
import traceback

usage = """

usage: twix2html [twixpath1] [twixpath2] ... [twixpathN]
"""

def twix2html_wrapper(*args):
    for twixpath in args:
        twix2html(twixpath, save=True)
        

def main():
    try:
        twixpath = sys.argv[1::]
        twix2html_wrapper(*twixpath)
    except Exception as exc:
        print(traceback.format_exc())
        print(exc)
        print(usage)



    

