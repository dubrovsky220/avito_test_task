import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import json

import torch
import torch.nn as nn

from torchcrf import CRF


RANDOM_STATE = 99

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
torch.cuda.manual_seed(RANDOM_STATE)
torch.cuda.manual_seed_all(RANDOM_STATE)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.embedding_dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.ln = nn.LayerNorm(hidden_dim*2)
        self.fc_dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim*2, 2)
        self.crf = CRF(num_tags=2, batch_first=True)


    def forward(self, x, tags=None, mask=None):
        """
        Args:
            x: [batch, seq_len] индексы символов
            tags: [batch, seq_len] метки (0/1), для обучения
            mask: [batch, seq_len] 1=реальный символ, 0=padding
        Returns:
            Если tags передан: loss
            Иначе: предсказанные метки
        """
        emb = self.embedding(x)  # [batch, seq_len, embed_dim]
        emb = self.embedding_dropout(emb)
        lstm_out, _ = self.lstm(emb)  # [batch, seq_len, hidden_dim*2]
        lstm_out = self.ln(lstm_out)
        lstm_out = self.fc_dropout(lstm_out)
        emissions = self.fc(lstm_out)  # [batch, seq_len, 2]

        if tags is not None:
            # обучение
            loss = -self.crf(emissions, tags, mask=mask, reduction='mean')
            return loss
        else:
            # инференс
            pred_tags = self.crf.decode(emissions, mask=mask)
            return pred_tags
        

def load_char2id(checkpoint_dir="./checkpoints", model_name="bilstm_space"):
    """
    Загружает словарь char2id.

    Args:
        model_params (dict): параметры для инициализации модели
        checkpoint_dir (str): директория с сохранёнными файлами
        model_name (str): имя модели (без расширения)

    Returns:
        char2id (dict): словарь char2id
    """
    checkpoint_dir = Path(checkpoint_dir)

    # Загружаем словарь
    with open(checkpoint_dir / f"{model_name}_char2id.json", "r", encoding="utf-8") as f:
        char2id = json.load(f)
    
    return char2id
        

def load_model(model_class, model_params, checkpoint_dir="./checkpoints", model_name="bilstm_space", device="cpu"):
    """
    Загружает сохранённую модель для инференса или дообучения.

    Args:
        model_class (nn.Module): класс модели
        model_params (dict): параметры для инициализации модели
        checkpoint_dir (str): директория с сохранёнными файлами
        model_name (str): имя модели (без расширения)
        device (str): устройство для загрузки ("cpu" или "cuda")

    Returns:
        model (nn.Module): восстановленная модель в режиме eval
    """

    checkpoint_dir = Path(checkpoint_dir)

    # Создаём модель и загружаем веса
    model = model_class(**model_params)
    state_dict = torch.load(checkpoint_dir / f"{model_name}.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"Модель {model_name} успешно загружена")

    return model


def text_to_tensor(text, char2id, max_len=128, device="cpu"):
    """
    Преобразует строку без пробелов в тензор индексов с паддингом.

    Args:
        text (str): строка текста без пробелов
        char2id (dict): словарь символов → индексы
        max_len (int): максимальная длина последовательности (усечение или паддинг)
        device (str): устройство ("cpu" или "cuda")

    Returns:
        torch.Tensor: тензор индексов формы (1, max_len)
    """
    
    ids = [char2id.get(ch, char2id["<UNK>"]) for ch in text]
    if len(ids) < max_len:
        ids = ids + [char2id["<PAD>"]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)


def restore_spaces(text: str, positions: list[int]) -> str:
    """
    Восстанавливает пробелы в тексте по предсказанным позициям.

    Args:
        text (str): строка без пробелов
        positions (list[int]): список индексов (1-based), после которых нужно вставить пробел

    Returns:
        str: текст с восстановленными пробелами
    """
    chars = list(text)
    result = []
    for i, ch in enumerate(chars, start=1):  # индексация с 1
        result.append(ch)
        if i in positions:
            result.append(" ")
    return "".join(result).strip()


def infer_file(model, char2id, input_path, output_path, restored_path=None, max_len=128, pad_idx=0, device="cpu"):
    """
    Выполняет инференс для набора строк и сохраняет предсказания в CSV.

    Args:
        model (nn.Module): обученная модель
        char2id (dict): словарь символов → индексы
        input_path (str | Path): путь к входному TXT файлу (id, text)
        output_path (str | Path): путь для сохранения результата
        restored_path (str | Path | None): путь для сохранения востановленных текстов (если задан)
        max_len (int): максимальная длина последовательности
        pad_idx (int): индекс паддинга
        device (str): устройство ("cpu" или "cuda")

    Returns:
        pd.DataFrame: таблица с предсказанными позициями пробелов
    """

    input_path = Path(input_path)
    output_path = Path(output_path)
    if restored_path:
        restored_path = Path(restored_path)

    results = []
    restored_texts = []

    model.eval()

    with open(input_path, "r", encoding="utf-8") as f:
        next(f)

        for line in f:
            line = line.strip()
            if not line:
                continue
            sample_id, text = line.split(",", 1)

            # Преобразуем текст в индексы
            x = text_to_tensor(text, char2id, max_len=max_len, device=device)

            with torch.no_grad():
                mask = (x != pad_idx).to(device)
                preds_list = model(x, mask=mask)  # decode через CRF
                preds = preds_list[0]             # единственный пример в батче

            positions = [i + 1 for i, label in enumerate(preds[:len(text)]) if label == 1]

            results.append({"id": sample_id, "predicted_positions": str(positions)})

            if restored_path:
                restored_texts.append({"id": sample_id, "restored_text": restore_spaces(text, positions)})

    # Сохраняем CSV с позициями
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Результат сохранён в {output_path}")

    # Сохраняем восстановленные тексты, если нужно
    if restored_path:
        df_restored = pd.DataFrame(restored_texts)
        df_restored.to_csv(restored_path, index=False)
        print(f"Восстановленные тексты сохранены в {restored_path}")

    return df


def get_device():
    """
    Определяет, какое устройство использовать для обучения/инференса.
    Если GPU доступна и совместима — возвращает cuda.
    Иначе fallback на CPU.
    """
    if torch.cuda.is_available():
        # Проверим compute capability текущего GPU
        major, minor = torch.cuda.get_device_capability()
        compute_capability = major + minor / 10
        # Минимальная поддерживаемая версия для установленного PyTorch
        min_supported = 7.0
        if compute_capability >= min_supported:
            print(f"Используется GPU: {torch.cuda.get_device_name(0)} (compute capability {major}.{minor})")
            return "cuda"
        else:
            print(f"GPU {torch.cuda.get_device_name(0)} (compute capability {major}.{minor}) не поддерживается, используется CPU")
            return "cpu"
    else:
        print("GPU недоступна, используется CPU")
        return "cpu"


def main():
    """
    Основная функция инференса модели BiLSTM-CRF для восстановления пробелов в текстах.

    Функция:
        1. Определяет доступное устройство (GPU/CPU) с помощью get_device().
        2. Загружает словарь символов (char2id) и параметры модели.
        3. Создает и загружает обученную модель BiLSTM-CRF.
        4. Выполняет инференс на входном файле TXT, предсказывает позиции пробелов.
        5. Сохраняет результаты в CSV с предсказанными позициями.
        6. Опционально сохраняет восстановленные тексты с пробелами в отдельный CSV.

    Аргументы командной строки:
        --input_path (str, обязательный): путь к входному текстовому файлу (id, текст без пробелов)
        --output_path (str, обязательный): путь для сохранения CSV с предсказанными позициями пробелов
        --restored_path (str, необязательный): путь для сохранения CSV с восстановленными текстами

    Использование:
        python main.py --input_path ./test.txt --output_path ./preds.csv --restored_path ./restored.csv
    """
    
    DEVICE = get_device()
    MODEL_DIRECTORY = "./model"
    MODEL_NAME = "bilstm_crf_space_v3"

    parser = argparse.ArgumentParser(description="Инференс модели BiLSTM-CRF для восстановления пробелов")
    parser.add_argument("--input_path", type=str, required=True, help="Путь к входному файлу (TXT)")
    parser.add_argument("--output_path", type=str, required=True, help="Путь для сохранения результатов (CSV)")
    parser.add_argument("--restored_path", type=str, required=False, help="Путь для сохранения восстановленных текстов (CSV, опционально)")
    args = parser.parse_args()

    char2id_loaded = load_char2id(MODEL_DIRECTORY, MODEL_NAME)

    model_params = {
        "vocab_size": len(char2id_loaded),
        "embed_dim": 128,
        "hidden_dim": 256,
        "num_layers": 3,
        "pad_idx": 0
    }

    model_loaded = load_model(
        model_class=BiLSTM_CRF,
        model_params=model_params,
        checkpoint_dir=MODEL_DIRECTORY,
        model_name=MODEL_NAME,
        device=DEVICE
    )

    df_results = infer_file(
        model=model_loaded,
        char2id=char2id_loaded,
        input_path=args.input_path,
        output_path=args.output_path,
        restored_path=args.restored_path,
        max_len=128,
        device=DEVICE
    )


if __name__ == "__main__":
    main()