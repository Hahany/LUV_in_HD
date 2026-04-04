# Deduplicate for the connected components
def Deduplicate(a):
    components = []
    starend_of_string = []
    stringlength = []
    for si, i in enumerate(a.connected_components):
        if i not in components:
                components.append(i)
                starend_of_string.append(a.starend_of_string[si])
                stringlength.append(a.stringlength[si])
    a.connected_components = components
    a.starend_of_string = starend_of_string
    a.stringlength = stringlength

def deepDeduplicate(a):
    components = []
    N = len(a.connected_components)
    for i in range(N):
        for j in range(i+1, N, 1):
            if set(a.connected_components[i]) == set(a.connected_components[j]):
                components.append(j)
    connected_components = []
    starend_of_string = []
    stringlength = []
    for i in range(N):
        if i not in components:
            connected_components.append(a.connected_components[i])
            starend_of_string.append(a.starend_of_string[i])
            stringlength.append(a.stringlength[i])
    a.connected_components = connected_components
    a.starend_of_string = starend_of_string
    a.stringlength = stringlength
