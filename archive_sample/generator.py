import pandas as pd
import numpy as np

date_range = pd.date_range(start="2023-03-01", end="2023-03-31", freq="D")

np.random.seed(42)
# Definir aqui o intervalo de valores
close_values = np.round(np.random.uniform(4, 7, size=len(date_range)), 2)

# Ajusta os fins de semana com o valor da sexta anterior
for i in range(len(date_range)):
    if date_range[i].weekday() in [5, 6]:
        j = i - 1
        while j >= 0 and date_range[j].weekday() != 4:
            j -= 1
        if j >= 0:
            close_values[i] = close_values[j]

# Open é igual ao valor de Close anterior
open_values = np.roll(close_values, 1)
open_values[0] = close_values[0]

df = pd.DataFrame({
    "Date": date_range,
    "Open": open_values,
    "Close Adj": close_values
})

# Definir aqui o nome do ficheiro
output_file = "APA.csv"
df.to_csv(output_file, index=False)