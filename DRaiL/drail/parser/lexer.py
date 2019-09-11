import ply.lex as lex


class Lexer(object):
    """
    Lexer to compile scripts
    """

    # list of token names
    tokens = [
            'IMPL',
            'COMMENT',
            'VARSTRING',
            'VARPRED',
            'VARINSTANCE',
            'VARARG',
            'NUMBER',
            'EQ',
            'LEQ',
            'GEQ',
            'SUMFORALL',
            'FLOAT',
            'GROUPARG'
            ]

    # reserved words
    reserved = {
            'entity': 'ENTITY',
            'predicate': 'PREDICATE',
            'dbmodule': 'DBMODULE',
            'dbclass': 'DBCLASS',
            'arguments': 'ARGUMENTS',
            'label': 'LABEL',
            'classes': 'LBCLASSES',
            'type': 'LBTYPE',
            'load': 'LOAD',
            'file': 'FILE',
            'rule': 'RULE',
            'conds': 'CONDS',
            'lambda': 'LAMBDA',
            'network': 'NETWORK',
            'ruleset': 'RULESET',
            'score': 'SCORE',
            'scmodule': 'SCMODULE',
            'scclass': 'SCCLASS',
            'fefunctions': 'FEFUNC',
            'feclass': 'FECLASS',
            'femodule': 'FEMODULE',
            'embedding': 'EMBEDDING',
            'vector': 'VECTOR',
            'input': 'INPUT',
            'dbfunction': 'DBFUNC',
            'hardconstr': 'HARDCONSTRAINT',
            'groupby': 'GROUPBY',
            'head': 'HEAD',
            'spliton': 'SPLITCLASSIF',
            'target': 'TARGET',
            'ArgumentType.String': 'ARGSTRING',
            'ArgumentType.Integer': 'ARGINT',
            'ArgumentType.Double': 'ARGDOUBLE',
            'ArgumentType.UniqueID': 'ARGID',
            'ArgumentType.UniqueString': 'ARGSTRINGID',
            'LabelType.Multiclass': 'LBMULTICLASS',
            'LabelType.Multilabel': 'LBMULTILABEL',
            'LabelType.Binary': 'LBBINARY',
            }

    # WARNING: I AM ASSUMING THERE ARE NO > OR < CONSTRAINTS!!!!! JUST <= OR >=
    # This is HACK for deadlines and has to be addressed
    literals = ['&','|', '~', ":", ",", ";", "(", ")", "[", "]", "{", "}", "^", "?",  '+', '*', '=', '.']

    # update tokens with reserved words
    tokens += ['ID'] + list(reserved.values())

    # Regular expression rules for simple tokens
    t_IMPL = r'=>'
    t_VARSTRING = r'".+?"'
    t_VARPRED = r'([A-Z]{1,1}[a-z]+)+'
    t_VARINSTANCE = r'[A-Z]{1,1}'
    t_VARARG = r'[a-z]+([A-Z]{1,1}[a-z]*)*'
    t_FLOAT = r'\d+\.\d+'
    t_NUMBER = r'\d+'
    t_GEQ = r'>='
    t_LEQ = r'<='
    t_EQ = r'=='
    t_SUMFORALL = r'\\sum_'
    t_GROUPARG = r'([A-Z]{1,1}[a-z]+)+\.\w+'

    # A string containing ignored characters (spaces and tabs)
    t_ignore = '\t \n'

    # Convert to literal
    def t_dot(self, t):
        r'\.'
        t.type = '.'
        return t

    # Convert to literal
    def t_star(self, t):
        r'\*'
        t.type = '*'
        return t

    # Convert to literal
    def t_plus(self, t):
        r'\+'
        t.type = '+'
        return t

    # Convert to literal
    def t_or(self, t):
        r'\|'
        t.type = '|'
        return t

    # Convert to literal
    def t_lparen(self, t):
        r'\('
        t.type = '('
        return t

    # Convert to literal
    def t_rparen(self, t):
        r'\)'
        t.type = ')'
        return t

    # Convert to literal
    def t_lbracket(self, t):
        r'\['
        t.type = '['
        return t

    # Convert to literal
    def t_rbracket(self, t):
        r'\]'
        t.type = ']'
        return t

     # Convert to literal
    def t_lbrace(self, t):
        r'\{'
        t.type = '{'
        return t

    # Convert to literal
    def t_rbrace(self, t):
        r'\}'
        t.type = '}'
        return t

    # Convert to literal
    def t_circumflex(self, t):
        r'\^'
        t.type = '^'
        return t

    # Convert to literal
    def t_interrogation(self, t):
        r'\?'
        t.type = '?'
        return t

    # Define a rule so we can track line numbers
    def t_newline(self, t):
        r'\n+'
        t.lexer.lineno += len(t.value)

    # Define a rule to track reserved words
    def t_ID(self, t):
        r'(entity|predicate|arguments|label|classes|type|load|file|ruleset|conds|lambda|network|rule|fefunctions|feclass|femodule|hardconstr|groupby|head|spliton|target|ArgumentType\.[A-Za-z]+|LabelType\.[A-Za-z]+|dbmodule|dbclass|dbfunction|embedding|vector|input|score|scmodule|scclass)'
        t.type = self.reserved.get(t.value,'ID')
        return t

    # Define a rule to discard comments
    def t_COMMENT(self, t):
        r'//.*'
        pass

    # Error handling rule
    def t_error(self, t):
        print("Illegal character '%s'" % t.value[0])
        t.lexer.skip(1)

    # Build the lexer
    def build(self, **kwargs):
        self.lexer = lex.lex(module=self, **kwargs)

    # Test its output by tokenizing
    def test(self, data):
        self.lexer.input(data)
        while True:
             tok = self.lexer.token()
             if not tok:
                 break
             print(tok)
