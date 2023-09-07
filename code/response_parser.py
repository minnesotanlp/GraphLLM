
def parse_response(response, delimiter):
    try:
        start_index = response.index(delimiter) + len(delimiter)
        value = response[start_index:].strip()
        if '-1' in value :
            return '-1' 
        elif '?' in value:
            return '?' # what if the response is just repeating the node value : ? -- DEBUG
        else:
            return value
    except ValueError:
        return None
