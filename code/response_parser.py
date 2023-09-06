
def parse_response(response, delimiter):
    try:
        start_index = response.index(delimiter) + len(delimiter)
        value = response[start_index:].strip()
        if '?' in value :
            return '?'
        else:
            return value
    except ValueError:
        return None
