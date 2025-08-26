import torch
import json
import pandas as pd
import logging as lg
from pathlib import Path
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          BitsAndBytesConfig)
from random import sample
# from sentence_transformers import SentenceTransformer, util

from typing import Dict, List

main_log = lg.getLogger(__name__)

# --- Rutas importantes ---
EXPERIMENTS_DIR = Path(__file__).resolve().parent

# ----- General -----


def check_path(path: str | Path) -> Path:
  """
  Revisa:
  1. Que una ruta sea del tipo correcto.
  2. Que exista.

  Args:
      path (str | Path): Ruta a revisar.

  Raises:
      TypeError: Si la ruta no es str o Path.
      FileNotFoundError: Si la ruta no existe.

  Returns:
      Path: Devuelve la ruta como objeto Path.
  """
  # Checamos que sea str o Path
  try:
    path = Path(path)
  except TypeError:
    msg = f"check_path: '{path}' tiene un formato inválido."
    main_log.error("🔴 " + msg)
    raise TypeError(msg)

  # Y ya vemos si existe o no
  if not path.exists():
    msg = f"check_path : '{path}' no encontrado."
    main_log.error(f"🔴 {msg}")
    raise FileNotFoundError(msg)

  return path


# ----- Para usar en NanoGCG -----


def get_dataset(dataset: str,
                random: bool,
                num_conv: int):

  # Verificamos que la opción de dataset sea válida
  dataset = dataset.lower()
  if dataset not in ["msc", "bst"]:
    raise Exception("Dataset '{dataset}' no reconocido.")

  # Elegimos la ruta correcta según el dataset indicado
  FILENAME_DATA = EXPERIMENTS_DIR / "_data" / "conv"
  if dataset == "msc":
    FILENAME_DATA /= "msc_30_short.json"
  else:
    FILENAME_DATA /= "bst_30_short.json"

  FILENAME_DATA = check_path(FILENAME_DATA)

  # Leemos el archivoi
  with open(FILENAME_DATA, "r") as f:
    raw_data = json.load(f)
    main_log.info(f"💬 Dataset cargado : {FILENAME_DATA.stem}")

    # Cargamos la cantidad de conversaciones solicitadas (¿de forma aleatoria?)
    if random:
      conversations = sample(raw_data, num_conv)
    else:
      conversations = raw_data[:num_conv]

  return conversations


def load_model_tokenizer(mod: str = "llama-2-7b-chat",
                         where: str = "cluster",
                         mod_prec: str = "full"):
  """
  Carga y devuelve el modelo y el tokenizador.

  Args:
      mod (str, optional): Nombre del modelo, debe coincidir con las "claves" del archivo _models_ids_json. Defaults to "llama-2-7b-chat".
      where (str, optional): Desde dónde se va a cargar el modelo (cluster, appa o huggingface). Defaults to "cluster".

  Returns:
      El modelo y el tokenizador.
  """
  # Evita imprimir barras de progreso al cargar el modelo y el tokenizador
  from transformers.utils import logging
  logging.disable_progress_bar()

  # Opciones de precisión
  bnb_8bit = BitsAndBytesConfig(load_in_8bit=True)
  bnb_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
  )
  precission = {
    "full": {"torch_dtype": torch.float32},
    "half": {"torch_dtype": torch.bfloat16},
    "8bit": {"quantization_config": bnb_8bit},
    "4bit": {"quantizatino_config": bnb_4bit}
  }

  # Verificamos que la opción de precisión sea correcta
  mod_prec = mod_prec.lower()
  if mod_prec not in precission.keys():
    raise ValueError(
      "load_model_tokenizer : Precisión '{mod_prec}' no válida."
    )

  # Cargamos el "directorio" de modelos
  with open(EXPERIMENTS_DIR / "_models_ids.json", "r") as f:
    models = json.load(f)

  # Verificamos que la opción del modelo sea válida
  mod = mod.lower()
  if mod not in models["cluster"].keys():
    raise Exception(
      "load_model_tokenizer : Modelo '{mod}' no reconocido."
    )

  # La ruta al modelo
  MODELS_DIR = check_path(models[where]["directorio"])
  model_id = MODELS_DIR / models[where][mod]

  # Cargamos el modelo y el tokenizador
  model = AutoModelForCausalLM.from_pretrained(
    model_id,
    **precission[mod_prec],
    local_files_only=True,
    device_map="auto",
  )
  tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    local_files_only=True,
  )

  main_log.info(f"💬 Dispositivo usado : {model.device}")
  main_log.info(f"💬 Precisión : {mod_prec}")
  main_log.info("🟢 Modelo cargado correctamente.")

  return model, tokenizer

# ----- Generación -----


def load_datasets(path: str | Path) -> Dict:
  """
  Carga todos los conjuntos de datos a un diccionario para tenerlos disponibles.

  Args:
      path (str | Path): Ruta al directorio con los conjuntos de datos.

  Returns:
      Dict: Diccionario con los conjuntos de datos.
  """
  path = check_path(path)

  datasets = {}
  for dataset in path.glob("*short*"):
    with open(path / dataset, "r") as f:
      datasets[dataset.stem[:3]] = (json.load(f))
  main_log.info("🟢 Datasets cargados correctamente")
  return datasets


def load_results(path: str | Path) -> List:
  path = check_path(path)

  # Como cada ejecución de experiment.py genera un único
  # directorio output, entonces solo hay un .jsonl en cada caso
  result_file = list(path.glob("*.jsonl"))[0]

  results = []

  main_log.info(result_file.name)
  with open(result_file, "r") as f:
    for line in f:
      tmp = json.loads(line)
      tmp["dataset"] = result_file.name[:3]
      tmp["file"] = result_file.stem
      results.append(tmp)
  main_log.info("🟢 Resultados cargados correctamente")
  return results


def generate(model,
             tokenizer,
             messages: list = None,
             adv_string: str = None,
             max_new_tokens: int = 32) -> str:

  if messages and adv_string:
    messages[0]['content'] = adv_string + ' ' + messages[0]['content']
  elif messages and (not adv_string):
    messages = messages
  elif (not messages) and adv_string:
    messages = [{'role': 'user', 'content': adv_string}]
  else:
    raise ValueError("generate : No hay mensaje ni cadena adversarial.")

  input = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
  ).to(model.device)
  output = model.generate(
    input,
    do_sample=False,
    max_new_tokens=max_new_tokens
  )
  output_str = tokenizer.batch_decode(
    output[:, input.shape[1]:],
    skip_special_tokens=True
  )[0]

  return output_str

# ----- Evaluación -----
