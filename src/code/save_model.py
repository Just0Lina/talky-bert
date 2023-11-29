from sentence_transformers import SentenceTransformer
import torch

# Инициализация модели
model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

# Сохранение модели
path_to_save = F'/content/gdrive/MyDrive/model2.pth'
torch.save(model, path_to_save)