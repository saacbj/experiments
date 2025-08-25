from parlai.core.params import ParlaiParser
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task

import json

# --- Parámetros
# Cuántas conversaciones se van a guardar
num_examples = 30
# Nombre del archivo para guardar las conversaciones
filename = "bst" + f"_{num_examples}"

# --- Configura ParlAI
parser = ParlaiParser()
parser.set_params(
    task='blended_skill_talk',
    datatype='train:ordered',
    batchsize=1
)
opt = parser.parse_args([])

# Agente dummy y entorno de diálogo
agent = RepeatLabelAgent(opt)
world = create_task(opt, agent)

# Contenedor de datos
examples = []

for i in range(num_examples):
  conversation = [i]
  eop = False
  first_message = True

  while (not eop):
    world.parley()
    eop = world.episode_done()
    msg = world.get_acts()[0]

    if first_message:
      msg_text = msg.get('text', '').split("\n")
      label = msg.get('labels', [''])[0]
      conversation.append({"role": "user", "content": msg_text[2]})
      conversation.append({"role": "assistant", "content": msg_text[3]})
      conversation.append({"role": "user", "content": msg_text[4]})
      conversation.append({"role": "assistant", "content": label})
      first_message = False
    else:
      conversation.append({"role": "user", "content": msg.get('text', '')})
      conversation.append(
        {"role": "assistant", "content": msg.get('labels', [''])[0]}
      )

  examples.append(conversation)

# --- Guardamos las conversaciones completas

with open(f'data/{filename}.json', 'w') as f_json:
  json.dump(examples, f_json, indent=2)

# --- Creamos el conjunto de las conversaciones cortas

bst_short = []
for conv in examples:
  curr_conv = [conv[0]]
  curr_conv += conv[-8:]
  bst_short.append(curr_conv)

filename += '_short'

with open(f'data/{filename}.json', 'w') as f_json:
  json.dump(bst_short, f_json, indent=2)
