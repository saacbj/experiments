import torch
import json
import pandas as pd
import logging
from bert_score import score
from pathlib import Path
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer)
from random import sample

from typing import Dict, List

log = logging.getLogger(__name__)

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
    log.error("🔴 " + msg)
    raise TypeError(msg)

  # Y ya vemos si existe o no
  if not path.exists():
    msg = f"check_path : '{path}' no encontrado."
    log.error(f"🔴 {msg}")
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
    log.info(f"💬 Dataset cargado : {FILENAME_DATA.stem}")

    # Cargamos la cantidad de conversaciones solicitadas (¿de forma aleatoria?)
    if random:
      conversations = sample(raw_data, num_conv)
    else:
      conversations = raw_data[:num_conv]

  return conversations


def load_model_tokenizer(mod: str = "llama-2-7b-chat",
                         where: str = "cluster"):
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

  # Cargamos el "directorio" de modelos
  with open(EXPERIMENTS_DIR / "_models_ids.json", "r") as f:
    models = json.load(f)

  # Verificamos que la opción del modelo sea válida
  mod = mod.lower()
  if mod not in models["cluster"].keys():
    raise Exception("load_model_tokenizer : Modelo '{mod}' no reconocido.")

  # La ruta al modelo
  MODELS_DIR = check_path(models[where]["directorio"])
  model_id = MODELS_DIR / models[where][mod]

  # Cargamos el modelo y el tokenizador
  # TODO : ¿Qué hacer con los parámetros de carga del tokenizador y el modelo?
  model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
    device_map="auto",
  )
  tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    local_files_only=True,
  )

  log.info(f"💬 Dispositivo usado : {model.device}")
  log.info("🟢 Modelo cargado correctamente.")

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
  log.info("🟢 Datasets cargados correctamente")
  return datasets


def load_results(path: str | Path) -> List:
  path = check_path(path)

  result_files = list(path.glob("*.jsonl"))

  log.info(f"{len(result_files)} archivo(s) encontrado(s):")

  results = []

  for file in result_files:
    log.info(file.name)
    with open(file, "r") as f:
      for line in f:
        tmp = json.loads(line)
        tmp["dataset"] = file.name[:3]
        tmp["file"] = file.stem
        results.append(tmp)
  log.info("🟢 Resultados cargados correctamente")
  return results

# ----- Evaluación -----


def safe_text(x):
  return "" if x is None else str(x).strip()


def evaluate_responses(results):
  # Listas para batch scoring
  comp2, nc2, gt = [], [], []
  comp3, nc3 = [], []

  rows = []
  for r in results:
    rows.append(r)  # copia original

    comp2.append(safe_text(r["compression_llama2"]))
    nc2.append(safe_text(r["no_compression_llama2"]))
    comp3.append(safe_text(r["compression_llama3"]))
    nc3.append(safe_text(r["no_compression_llama3"]))
    gt.append(safe_text(r["ground_truth"]))

  # ----- Llama2 -----

  # comp vs nc
  _, _, f1_c_nc_2 = score(
    comp2,
    nc2,
    lang="en",
    model_type="roberta-large",
    rescale_with_baseline=True,
    verbose=False
  )

  # comp vs GT
  _, _, f1_c_gt_2 = score(
    comp2,
    gt,
    lang="en",
    model_type="roberta-large",
    rescale_with_baseline=True,
    verbose=False
  )

  # nc vs GT (se usa como baseline)
  _, _, f1_nc_gt_2 = score(
    nc2,
    gt,
    lang="en",
    model_type="roberta-large",
    rescale_with_baseline=True,
    verbose=False
  )

  # ----- Llama3 -----
  _, _, f1_c_nc_3 = score(
    comp3,
    nc3,
    lang="en",
    model_type="roberta-large",
    rescale_with_baseline=True,
    verbose=False
  )
  _, _, f1_c_gt_3 = score(
    comp3,
    gt,
    lang="en",
    model_type="roberta-large",
    rescale_with_baseline=True,
    verbose=False
  )
  _, _, f1_nc_gt_3 = score(
    nc3,
    gt,
    lang="en",
    model_type="roberta-large",
    rescale_with_baseline=True,
    verbose=False
  )

  # ----- Guardamos los resultados :)) -----
  out = []
  for i, r in enumerate(rows):
    cur = r.copy()
    cur["bert_c_nc_llama2"] = float(f1_c_nc_2[i])
    cur["bert_c_gt_llama2"] = float(f1_c_gt_2[i])
    cur["bert_nc_gt_llama2"] = float(f1_nc_gt_2[i])
    cur["bert_delta_llama2"] = cur["bert_c_gt_llama2"] - \
        cur["bert_nc_gt_llama2"]

    cur["bert_c_nc_llama3"] = float(f1_c_nc_3[i])
    cur["bert_c_gt_llama3"] = float(f1_c_gt_3[i])
    cur["bert_nc_gt_llama3"] = float(f1_nc_gt_3[i])
    cur["bert_delta_llama3"] = cur["bert_c_gt_llama3"] - \
        cur["bert_nc_gt_llama3"]
    out.append(cur)
  return out
