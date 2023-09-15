"""
def parse_response(response, delimiter):
    if response == '-1':
        return -1
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
"""
def parse_response(response, delimiter):
    try:
        if response == '-1':
            return '-1'
        
        if delimiter in response:
            start_index = response.index(delimiter) + len(delimiter)
            value = response[start_index:].strip()
            
            if '?' in value:
                return '?'
            elif '-1' in value:
                return '-1'
            else:
                return value
        else:
            if '?' in response:
                return '?'
            elif '-1' in response:
                return -1
            else:
                return response.strip()
    except ValueError:
        return None

# Test cases
"""
response1 = "-1"
response2 = "12345"
response3 = "Label of Node=1692 is 3"
response4 = "Node 2204 is also 3"
response5 = "?"
response6 ="Label of Node=1692 is -1"

delimiter = "="

print(parse_response(response1, delimiter))  # Output: '-1'
print(parse_response(response2, delimiter))  # Output: '12345'
print(parse_response(response3, delimiter))  # Output: '1692 is 3'
print(parse_response(response4, delimiter))  # Output: 'Node 2204 is also 3'
print(parse_response(response5, delimiter))  # Output: '?'
print(parse_response(response6, delimiter)) # Output : -1

"""



