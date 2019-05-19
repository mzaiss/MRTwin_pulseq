# Generated from XProtParser.g4 by ANTLR 4.7.1
from antlr4 import *
import re
if __name__ is not None and "." in __name__:
    from .XProtParser import XProtParser
else:
    from XProtParser import XProtParser

# This class defines a complete generic visitor for a parse tree produced by XProtParser.

class mstruct(dict):
    def __missing__(self, key):
        self[key] = mstruct()
        value = self[key]
        return value

def convert_line(line):
    line = str(line)
    newline = re.sub('\.([a-zA-Z_0-9]+)(?=.*\=)',lambda matchobj: "['" + matchobj.group(1) + "']", line)    
    return newline

def strip_quotes(s):
    try:
        if s[0]=='"' and s[-1]=='"':
            s = s[1:-1]
    except:
        return s
    
    return s

def parse_ascconv(buf):
    header_dict = mstruct()
    header_entries =  buf.splitlines()
    for line in header_entries:
        newline = convert_line(line)
        newline2 = re.sub('^([a-zA-Z_0-9]+)(?=.*\=)', lambda matchobj: 
        "header_dict['" + matchobj.group(1) + "']", newline, count=1)
        #print('{}'.format(newline2))
        try:
            exec(str(newline2))
        except:
            pass

    return header_dict

class XProtParserVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by XProtParser#header.
    def visitHeader(self, ctx:XProtParser.HeaderContext):
        val  = dict()
        xprot   = []
        ascconv = []

        for xp in ctx.xprot():
            xprot.append(self.visit(xp))

        for asc in ctx.ASCCONV_STRING():
            asc_dict = parse_ascconv(asc.getText())
            ascconv.append(asc_dict)

        if len(xprot) == 1:
            xprot = xprot[0]

        if len(ascconv) == 1:
            ascconv = ascconv[0]


        val['xprot'] = xprot
        val['ascconv'] = ascconv

        return val

    # Visit a parse tree produced by XProtParser#xprot.
    def visitXprot(self, ctx:XProtParser.XprotContext):
        val = {}
        
        for c in ctx.special_node():
            (ckey,cval) = self.visit(c)
            val[ckey] = cval

        for c in ctx.node():
            (ckey,cval) = self.visit(c)
            val[ckey] = cval

        return val
            

    # Visit a parse tree produced by XProtParser#xprot_tag.
    def visitXprot_tag(self, ctx:XProtParser.Xprot_tagContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by XProtParser#special_node.
    def visitSpecial_node(self, ctx:XProtParser.Special_nodeContext):
        
        return self.visit(ctx.children[0])

    # Visit a parse tree produced by XProtParser#node.
    def visitNode(self, ctx:XProtParser.NodeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by XProtParser#param_special.
    def visitParam_special(self, ctx:XProtParser.Param_specialContext):
        key = self.visit(ctx.param_special_tag())
        val = eval(ctx.children[1].getText())
        return key,val


    # Visit a parse tree produced by XProtParser#param_special_tag.
    def visitParam_special_tag(self, ctx:XProtParser.Param_special_tagContext):
        return strip_quotes(ctx.SPECIAL_TAG_TYPE().getText())


    # Visit a parse tree produced by XProtParser#param_eva.
    def visitParam_eva(self, ctx:XProtParser.Param_evaContext):
        key = self.visit(ctx.param_eva_tag())
        val = strip_quotes(ctx.STRING_TABLE().getText())
        return key,val

    # Visit a parse tree produced by XProtParser#param_eva_tag.
    def visitParam_eva_tag(self, ctx:XProtParser.Param_eva_tagContext):
        key = strip_quotes(ctx.EVASTRINGTABLE().getText())
        return key


    # Visit a parse tree produced by XProtParser#param_array.
    def visitParam_array(self, ctx:XProtParser.Param_arrayContext):
        key = self.visit(ctx.param_array_tag())
        val = dict()
        val['default'] = self.visit(ctx.node())
        arr = []
        for c in ctx.array_value():
            arr.append(self.visit(c))

        val['data'] = arr

        return key,val
        

    # Visit a parse tree produced by XProtParser#param_array_tag.
    def visitParam_array_tag(self, ctx:XProtParser.Param_array_tagContext):
        return strip_quotes(ctx.TAG_NAME().getText())

    # Visit a parse tree produced by XProtParser#param_map.
    def visitParam_map(self, ctx:XProtParser.Param_mapContext):
        key = self.visit(ctx.param_map_tag())
        val = dict()
        for c in ctx.node():
            (ckey, cval) = self.visit(c)
            val[ckey] = cval
        
        return key,val


    # Visit a parse tree produced by XProtParser#param_map_tag.
    def visitParam_map_tag(self, ctx:XProtParser.Param_map_tagContext):
        return strip_quotes(ctx.TAG_NAME().getText())


    # Visit a parse tree produced by XProtParser#array_value.
    def visitArray_value(self, ctx:XProtParser.Array_valueContext):
        val = []
        for c in ctx.arr_val_item():
            val.append(self.visit(c))

        return val


    # Visit a parse tree produced by XProtParser#String.
    def visitString(self, ctx:XProtParser.StringContext):
        return ctx.getText()[1:-1]


    # Visit a parse tree produced by XProtParser#Number.
    def visitNumber(self, ctx:XProtParser.NumberContext):
        return float(ctx.getText())


    # Visit a parse tree produced by XProtParser#AnArrayVal.
    def visitAnArrayVal(self, ctx:XProtParser.AnArrayValContext):
        val = []
        for c in ctx.array_value().arr_val_item():
            val.append(self.visit(c))


    # Visit a parse tree produced by XProtParser#param_generic.
    def visitParam_generic(self, ctx:XProtParser.Param_genericContext):
        key = self.visit(ctx.param_generic_tag())
        val = []
        for c in ctx.param_generic_val():
            val.append(self.visit(c))
        
        return key,val

    # Visit a parse tree produced by XProtParser#param_generic_val.
    def visitParam_generic_val(self, ctx:XProtParser.Param_generic_valContext):
        val = ctx.children[0].getText()
        try: 
            val = eval(val)
        except:
            pass
        
        val = strip_quotes(val)

        return val
            

    # Visit a parse tree produced by XProtParser#param_generic_tag.
    def visitParam_generic_tag(self, ctx:XProtParser.Param_generic_tagContext):
        tag_type = strip_quotes(ctx.TAG_TYPE().getText())
        if hasattr(ctx,'TAG_NAME'):
            tag_type = strip_quotes(ctx.TAG_NAME().getText())

        return tag_type





del XProtParser

