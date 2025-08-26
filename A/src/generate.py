import typer
import json
import sys
import uuid
import gc
import logging as log
import time
from datetime import datetime
from pathlib import Path


# Cargamos rutas a directorios de importancia
EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent.parent
SCHEME_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = SCHEME_DIR / "results"

# Importamos nuestro módulo utils.py
sys.path.append(str(EXPERIMENTS_DIR))
from utils import *  # noqa: E402


def main(output_dir: str,
         mod: str = "llama-2-7b-chat"):

  # --- Cremos el directorio para guardar la generación ---
  GENERATION_DIR = RESULTS_DIR / output_dir / "generation"
  GENERATION_DIR.mkdir(exist_ok=True)
  hash = uuid.uuid4().hex[:4]
  filename_mod = '_'.join(mod.split('-')[:3])
  filename_mod = filename_mod.replace('.', '')
  FILENAME_OUT = GENERATION_DIR / f"{filename_mod}_{hash}"

  # Escribimos metadatos
  # ¿Qué metadatos necesitamos?
  with open(FILENAME_OUT.with_suffix('.jsonl'), "a") as f:
    json.dump({'from': output_dir,
               'model': mod,
               'hash': hash,
               'date': datetime.now().strftime("%Y %m %d %H:%M:%S")},
              f,
              ensure_ascii=False)
    f.write("\n")

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

  # ***** ***** ******

  # --- Cargamos los conjuntos de datos ---
  DATASETS_DIR = EXPERIMENTS_DIR / "_data" / "conv"
  datasets = load_datasets(DATASETS_DIR)

  # --- Y los resultados en su formato jsonl ---
  OUTPUT_DIR = SCHEME_DIR / "results" / output_dir
  results = load_results(OUTPUT_DIR)

  # --- Cargamos el modelo y el tokenizador ---
  model, tokenizer = load_model_tokenizer(mod)

  # --- Generamos texto ---
  for i, result in enumerate(results):
    _start = time.time()

    log.info(f"⚪️ Conversación {i + 1}/{len(results)}:")

    chat_id = result['id']
    adv_suffix = result['best_string']
    conversation = datasets[result['dataset']][result['id']][1:]
    ground_truth = conversation[-1]['content']
    curr = {
      'id': chat_id,
      'adv_suffix': adv_suffix,
      'conversation': conversation,
      'ground_truth': ground_truth,
      'generation': dict()
    }

    # La versión sin comprimir
    curr["generation"]["no_compression"] = generate(
      model,
      tokenizer,
      conversation[:-1],
    )
    log.info("\t🟢 Sin comprimir")

    # La versión comprimida
    curr["generation"]["compression"] = generate(
      model,
      tokenizer,
      conversation[4:-1],
      adv_suffix,
    )
    log.info("\t🟢 Comprimida")

    # Solo la cadena adversarial
    curr["generation"]["adv_string_only"] = generate(
      model,
      tokenizer,
      adv_string=adv_suffix,
    )
    log.info("\t🟢 Sólo la cadena adversarial")

    # Cadenas random
    curr["generation"]["random"] = dict()
    for i in range(5):
      random_adv_suffix = ''.join(sample(
        list(tokenizer.get_vocab().keys()), 20)
      )
      curr["generation"]["random"][i] = {
        "adv_suffix": random_adv_suffix,
        "text": generate(
          model,
          tokenizer,
          conversation[4:-1],
          random_adv_suffix
        )
      }
    log.info("\t🟢 Cadenas aleatorias")

    # Guardamos en el jsonl
    with open(FILENAME_OUT.with_suffix('.jsonl'), "a") as f:
      json.dump(curr, f, ensure_ascii=False)
      f.write("\n")

    log.info(
      f"🟢 Conversación {i + 1}/{len(results)} terminada. [{(time.time() - _start) / 60} min]"
    )

  # --- Liberamos el espacio ---
  del model
  gc.collect()
  torch.cuda.empty_cache()


if __name__ == "__main__":
  typer.run(main)
