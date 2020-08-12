import pandas as pd
import numpy as np

s = pd.Series(list('abca'))
print(s)

print(pd.get_dummies(s))

s1 = ['a', 'b', np.nan]
print(s1)
print(pd.get_dummies(s1))
print(pd.get_dummies(s1, dummy_na=True))

df = pd.DataFrame({'A': ['a', 'b', 'a'], 'B': ['b', 'a', 'c'], 'C': [1, 2, 3]})
print(df)
print(pd.get_dummies(df))
