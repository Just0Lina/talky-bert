import time
import torch
import numpy as np
from sentence_transformers import util

class CommandRecognizer:
    def __init__(self, model=None, arr=None):
        self.model = self.init_model(model)
        self.arr = self.init_array(arr)

    def recognize_command(self, sent):
        start_time = time.time()

        embedding_new = self.model.encode(sent, convert_to_tensor=False)

        command = "result['command']"
        maxerr = 100
        for row in self.arr:
            center_of_mass = row[0:-1]
            center = center_of_mass.astype(np.float32)
            errors_new = 1 - util.cos_sim(embedding_new, center)
            if errors_new <= maxerr:
                maxerr = errors_new
                command = row[-1]

        print(f"Time taken for recognition: {time.time() - start_time} seconds")
        print(sent, command)
        return command

    def init_model(self, model):
        if model is None:
          path = F'/content/gdrive/MyDrive/model.pth'
          model = torch.load(path)
          # model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        return model

    def init_array(self, arr):
        if arr is None:
            path = F"/content/gdrive/MyDrive/results.csv"
            arr = np.loadtxt(path, delimiter=',')
        return arr

recognizer = CommandRecognizer()
print(recognizer.recognize_command("Включи тайме"))
print(recognizer.recognize_command("Перезапусти"))
print(recognizer.recognize_command("Назад хочу"))
