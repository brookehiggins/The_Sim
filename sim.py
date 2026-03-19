import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data.csv')

# removes 'date' column if it exists, since it's not needed for the simulation
if 'date' in df.columns:
    df = df.drop(columns=['date'])

# if CSV has an unnamed index column, removes it
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

labels = df.columns.tolist()

history = df.to_numpy().tolist()
state = df.iloc[-1].to_numpy()

TIME_STEPS = 27

weights = np.array([0.2, 0.2, 0.2, 0.1, 0.12, 0.18])

influence = np.array([
    [0,    0.1,  0.1,  0.05, 0.1,  0.1],
    [0.05, 0,    0.1,  0.05, 0.05, 0.1],
    [0.1,  0.1,  0,    0.1,  0.05, 0.1],
    [0.05, 0.05, 0.1,  0,    0.05, 0.1],
    [0.1,  0.05, 0.05, 0.05, 0,    0.1],
    [0.1,  0.1,  0.1,  0.1,  0.1,  0]
])

MIN_VALUE = 0
MAX_VALUE = 10

wellbeing_over_time = []

# wellbeing for real data
for row in history:
    wellbeing_over_time.append(np.dot(weights, np.array(row)))

# simulate forward
for t in range(TIME_STEPS):
    if t == 14:
        state[0] -= 1.5
        state[4] -= 1.2
        state[5] -= 1.0

    change = np.random.uniform(-0.5, 0.5, size=state.shape)
    interaction = influence @ (state - 5) * 0.05
    baseline_pull = 0.05 * (5 - state)
    state = state + change + interaction + baseline_pull
    state = np.clip(state, MIN_VALUE, MAX_VALUE)

    history.append(state.copy().tolist())
    wellbeing_over_time.append(np.dot(weights, state))

history = np.array(history)

plt.figure(figsize=(11, 6))

for i in range(history.shape[1]):
    plt.plot(history[:, i], linestyle='--', marker='o', label=labels[i])

plt.plot(wellbeing_over_time, linewidth=3, label='Total Wellbeing')

plt.legend()
plt.axvline(x=len(df), linestyle='--', label='Simulation Start')
plt.title('The Sim v0.1 — Real Data + Simulated Future')
plt.xlabel('Time Steps')
plt.ylabel('Wellbeing')
plt.grid()
plt.show()