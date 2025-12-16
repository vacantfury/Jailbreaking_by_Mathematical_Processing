import pandas as pd

splits = {'mini_JailBreakV_28K': 'JailBreakV_28K/mini_JailBreakV_28K.csv', 'JailBreakV_28K': 'JailBreakV_28K/JailBreakV_28K.csv'}
df = pd.read_csv("hf://datasets/JailbreakV-28K/JailBreakV-28k/" + splits["mini_JailBreakV_28K"])

import pandas as pd

df = pd.read_csv("hf://datasets/JailbreakV-28K/JailBreakV-28k/JailBreakV_28K/RedTeam_2K.csv")