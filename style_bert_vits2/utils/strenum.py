import enum


class StrEnum(str, enum.Enum):
    """
    Enum where members are also (and must be) strings (backported from Python 3.11).
    """

    def __new__(cls, *values: str) -> "StrEnum":
        "values must already be of type `str`"
        if len(values) > 3:
            raise TypeError("too many arguments for str(): %r" % (values,))
        if len(values) == 1:
            # it must be a string
            if not isinstance(values[0], str):  # type: ignore
                raise TypeError("%r is not a string" % (values[0],))
        if len(values) >= 2:
            # check that encoding argument is a string
            if not isinstance(values[1], str):  # type: ignore
                raise TypeError("encoding must be a string, not %r" % (values[1],))
        if len(values) == 3:
            # check that errors argument is a string
            if not isinstance(values[2], str):  # type: ignore
                raise TypeError("errors must be a string, not %r" % (values[2]))
        value = str(*values)
        member = str.__new__(cls, value)
        member._value_ = value
        return member

    @staticmethod
    def _generate_next_value_(
        name: str, start: int, count: int, last_values: list[str]
    ) -> str:
        """
        Return the lower-cased version of the member name.
        """
        return name.lower()
