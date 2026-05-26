# Deduplicate for the connected components
def Deduplicate(a):
    seen = set()
    components = []
    starend_of_string = []
    stringlength = []
    for si, i in enumerate(a.connected_components):
        key = tuple(i)
        if key not in seen:
            seen.add(key)
            components.append(i)
            starend_of_string.append(a.starend_of_string[si])
            stringlength.append(a.stringlength[si])
    a.connected_components = components
    a.starend_of_string = starend_of_string
    a.stringlength = stringlength
def deepDeduplicate(a):
    seen = set()
    connected_components = []
    starend_of_string = []
    stringlength = []
    for i, comp in enumerate(a.connected_components):
        key = frozenset(comp)  # frozenset 可哈希，直接用 set 去重
        if key not in seen:
            seen.add(key)
            connected_components.append(comp)
            starend_of_string.append(a.starend_of_string[i])
            stringlength.append(a.stringlength[i])
    a.connected_components = connected_components
    a.starend_of_string = starend_of_string
    a.stringlength = stringlength
