import sys
import os
from drail.parser import lexer


def main():

    filename = sys.argv[1]
    text = open(filename).read()
    # Uncomment to text lexer alone
    lex = lexer.Lexer()
    lex.build()

    lex.test(text)

if __name__ == "__main__":
    main()
