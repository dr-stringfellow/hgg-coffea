import awkward as ak

constpt = ak.Array([[250,6,232,44,335,13,3,5,45,54,763,345,6,24,31,34],[88,77,66,43,12,345453,332,141,345576]])
constidx = ak.Array([[0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1],[0,0,0,0,0,0,0,1,1]])

counts = ak.run_lengths(constidx)
nested = ak.unflatten(constpt, ak.flatten(counts), axis=1)
print(nested)
what_I_want = ak.Array([[[250,6,232,44,335,13,3,5,45],[54,763,345,6,24,31,34]],[[88,77,66,43,12,345453,332],[141,345576]]])

