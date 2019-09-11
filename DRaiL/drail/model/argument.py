class Argument(object):

    def __init__(self, arg, isconstant, isobs=True, label=None):
        self.arg = arg
        self.isconstant = isconstant
        self.isobs = isobs
        self.label = label

    def __eq__(self, other):
        return self.arg == other.arg and \
               self.isconstant == other.isconstant

    def __neq__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        if not self.isconstant and self.isobs:
            return str(self.arg)
        elif self.isconstant and self.isobs:
            return '"' + str(self.arg) + '"'
        else:
            return str(self.arg) + "^" + str(self.label)

    def __repr__(self):
        return str(self)

class ArgumentDefinition(object):

    def __init__(self, name, typ):
        """
        Creates an argument definition composed by an
        argument name and its type. This class is used
        for the grounding procedure only.

        Args:
            name: argument name
            typ: argument type
        """
        self.name = name
        self.typ = typ

    def sqlite_type(self):
        """
        Get the sqlite3 equivalent type for an argument.
        It is used to create argument columns in the
        database.

        Return:
            A sqlite3 valid equivalent type
        """
        if self.typ == ArgumentType.String:
            return "TEXT"
        elif self.typ == ArgumentType.Integer:
            return "INTEGER"
        elif self.typ == ArgumentType.Double:
            return "REAL"
        elif self.typ == ArgumentType.UniqueID:
            return "INTEGER"
        elif self.typ == ArgumentType.UniqueString:
            return "TEXT"

    def __eq__(self, other):
        """
        Check if `other` argument is equal

        Args:
            other: argument to compare

        Returns:
            True if they are equal, False otherwise
        """
        return self.name == other.name and \
               self.typ == other.typ

    def __neq__(self, other):
        """
        Check if `other` argument is not equal

        Args:
            other: argument to compare

        Returns:
            True if they are not equal, False otherwise
        """
        return not self.__eq__(other)

    def __str__(self):
        """
        Obtain a string representation of an argument of the form:
        argument::argumentType

        Returns:
            A string representation of the argument
        """
        ret = self.name + "::"
        if self.typ == ArgumentType.String:
            ret += "ArgumentType.String"
        elif self.typ == ArgumentType.Integer:
            ret += "ArgumentType.Integer"
        elif self.typ == ArgumentType.Double:
            ret += "ArgumentType.Double"
        elif self.typ == ArgumentType.UniqueID:
            ret += "ArgumentType.UniqueID"
        elif self.typ == ArgumentType.UniqueString:
            ret += "ArgumentType.UniqueString"
        return ret

    def __repr__(self):
        """
        Use the string representation for argument when
        printing collections of arguments

        Returns:
            A string representation of the argument
        """
        return str(self)

class ArgumentType(object):
    """
    Enum type for an argument type: String, Integer, Double, UniqueID or UniqueString
    """
    String, Integer, Double, UniqueID, UniqueString = range(1, 6)

def sqlite_type(typ):
    """
    Get the sqlite3 equivalent type for an argument.
    It is used to create argument columns in the
    database.

    Return:
        A sqlite3 valid equivalent type
    """
    if typ == ArgumentType.String:
        return "TEXT"
    elif typ == ArgumentType.Integer:
        return "INTEGER"
    elif typ == ArgumentType.Double:
        return "REAL"
    elif typ == ArgumentType.UniqueID:
        return "INTEGER"
