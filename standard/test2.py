import awkward as ak
import numpy as np

constpt = ak.Array([[250,6,232,44,335,13,3,5,45,54,763,345,6,24,31,34],[88,77,66,43,12,345453,332,141,345576]])
constidx = ak.Array([[0,4,2,12],[5,2,4]])

print(constpt)

c = np.take(constpt,constidx)
