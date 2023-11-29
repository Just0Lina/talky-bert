import numpy as np
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import io
import csv


# Загрузка предварительно обученной модели
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# get data
data =pd.read_csv('../recourses/data_with_commands.csv', names=['Label', 'Command'], header=None, engine='python', sep=',')
command_labels = {}

# Iterate through the DataFrame rows
for index, row in data.iterrows():
    command = row['Command']
    label = row['Label']

    # Check if the command exists in the dictionary
    if command in command_labels:
        command_labels[command].append(label)
    else:
        # If the command doesn't exist, create a new key-value pair
        command_labels[command] = [label]
results =[]
# Display commands and their associated labels
for command, labels in command_labels.items():
    # print(f"Command: {command}, Labels: {', '.join(labels)}") # prints info
    embeddings = model.encode(labels, convert_to_tensor=False)


    # # Вычисление центра масс векторов Вторая реализация
    # center_of_mass = np.mean(embeddings, axis=0)
    # errors = 1 - util.cos_sim(embeddings, center_of_mass)
    # for index, emb in enumerate(embeddings):
    #   # results.append({
    #   #     # 'sentence': labels[index],
    #   #     'command': command,
    #   #     'center_of_mass': emb,
    #   #     # 'error': max(errors)
    #   # })
    #


    for index, emb in enumerate(embeddings):
      if (command != 'Command'):
        results.append(np.append(emb, int(command))) # добавили команду в конец для более удобного хранения

print(np.array(results))
path = F"/content/gdrive/MyDrive/results.csv"
np.savetxt("results.csv", np.array(results), delimiter=",")