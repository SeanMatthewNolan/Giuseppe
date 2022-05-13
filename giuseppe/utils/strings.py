def justify_str(_str: str, _len: int) -> str:
    """
    Fit string to set length
    Centers smaller strings
    Trims larger strings

    Parameters
    ----------
    _str: str
        string to justify
    _len: int
        length of output string

    Returns
    -------

    """
    if _len < 0:
        raise ValueError('Specified length should be greater than 0')

    if len(_str) <= _len:
        return _str.center(_len)
    else:
        return _str[:_len - 3] + '...'
