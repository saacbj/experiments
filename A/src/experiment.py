import nanogcg
import torch
import json
import copy
import typer
import gc
import time
import sys
from pathlib import Path
from tabulate import tabulate
from datetime import datetime
from nanogcg import GCGConfig

import logging as log  # logging

"""
Agosto 2025

Código pensado para correr pruebas del esquema A en el clúster de CIMAT.

La generación de texto se traslada a otro script dentro de la misma carpeta para evitar sobrecargar la VRAM. Lo mismo para la evaluación.

Se crea una carpeta output por cada ejecución (cada configuración) y dos archivos:

El output del código es :
- id de conversación (para después conectar con el mensaje)
- target
- best_string
- best_loss
- strings
- losses
"""

# --- Cargamos rutas a directorios de importancia ---
EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent.parent
SCHEME_DIR = Path(__file__).resolve().parent.parent

# --- Importamos el módulo utils.py ---
sys.append.path(EXPERIMENTS_DIR.as_posix())
from utils import *  # noqa: E402

# --- Creamos directorio OUT ---
# ../results/out_250810_144416
tstamp = datetime.now().strftime("%y%m%d_%H%M%S")
DIRECTORY_OUT = SCHEME_DIR / f"results/out_{tstamp}"
DIRECTORY_OUT.mkdir(parents=True, exist_ok=False)

# ----- Función main -----


def main(
  dataset: str = "msc",  # Conjunto de datos a usar (msc ó bst)
  num_steps: int = 250,  # Número de iteraciones
  topk: int = 64,  # Número de sustituciones de candidatos a considerar en una cierta posición
  search_width: int = 64,  # Número de candidatos en cada iteración de GCG
  seed: int = 42,  # Para reproductibilidad
  num_conv: int = 5,  # Cuántas conversaciones se van a usar por dataset
  random: bool = False,  # ¿Se toman aleatoriamente?
  mod: str = "llama-2-7b-chat"  # Qué modelo se usa en nanogcg
):

  # Nombre del archivo para el log y el csv
  # e.g. msc_250_64_32_5
  # hasta este momento, es el nombre sin sufijos
  tmstp = f"{dataset}_{num_steps}_{topk}_{search_width}_{num_conv}".lower()
  FILENAME_OUT = DIRECTORY_OUT / tmstp
  if FILENAME_OUT.exists():
    tmstp += "_{uuid.uuid4().hex[:4]}"
    FILENAME_OUT = DIRECTORY_OUT / tmstp
  del tmstp

  # ***** CONFIGURACIÓN DEL LOGGING *****

  # Formato del logging
  log.basicConfig(
    filename=FILENAME_OUT.with_suffix(".log"),
    filemode="w",  # "a" para no sobrescribir
    format="%(asctime)s - %(message)s",
    level=log.INFO,
    datefmt="%Y %m %d %H:%M:%S"
  )

  # Mensaje inicial
  log.info(f"{FILENAME_OUT.stem}")

  # ***** CÓDIGO PRINCIPAL *****

  # ----- Configuramos nanogcg -----

  # Configuración del GCG
  config = GCGConfig(
    num_steps=num_steps,
    search_width=search_width,
    topk=topk,
    seed=seed,
    verbosity="ERROR"
  )

  # Mostramos los parámetros en el log
  parameters = {
    "dataset": dataset,
    "num_steps": num_steps,
    "topk": topk,
    "search_width": search_width,
    "seed": seed,
    "num_conv": num_conv,
    "random": random,
    "model": mod
  }
  table = tabulate(
    [parameters],
    tablefmt="simple",
    headers="keys",
    showindex=False,
    numalign="center",
    stralign="center"
  )
  log.info(f"\n{table}")

  # ----- Cargamos modelos y tokenizadores -----

  model, tokenizer = load_model_tokenizer(mod)

  # ----- Datos -----

  # Leemos el contenido del dataset
  conversations = get_dataset(dataset, random, num_conv)

  for conversation in conversations:
    _start_time = time.time()  # Tiempo de ejecución

    # Copiamos el mensaje original
    base_message = copy.deepcopy(conversation[5:-1])

    # Quitamos los últimos 5 por el token de EOS
    target = tokenizer.apply_chat_template(
      conversation[1:5],
      tokenize=False
    )[:-5]

    # Añadimos el posicionador del adv_string
    # ❓ ¿Con o sin espacio?
    message = copy.deepcopy(base_message)
    message[0]["content"] = '{optim_str} ' + message[0]["content"]

    # ----- Ejecutamos el algoritmo -----

    result = nanogcg.run(
      model,
      tokenizer,
      message,
      target,
      config
    )

    # ----- Juntamos la información -----
    curr_result = {"id": conversation[0]}  # id de conversación
    curr_result["target"] = target
    curr_result["best_string"] = result.best_string
    curr_result["best_loss"] = result.best_loss
    curr_result["strings"] = result.strings
    curr_result["losses"] = result.losses

    # --- Guardamos los resultados ---

    with open(FILENAME_OUT.with_suffix(".jsonl"), "a") as f_json:
      json.dump(curr_result, f_json, ensure_ascii=False)
      f_json.write("\n")

    log.info(
      f"🟢 Conversación {conversation[0]} completada. [{(time.time() - _start_time) / 60}min]"
    )

  # ----- Liberamos espacio -----

  del tokenizer, model
  gc.collect()
  torch.cuda.empty_cache()

  log.info("💬 Espacio liberado.")


if __name__ == "__main__":
  typer.run(main)
