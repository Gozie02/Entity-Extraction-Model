#Import relevant packages
import pandas as pd
import spacy
from spacy.util import minibatch, compounding
import random
from spacy.training import Example
from pathlib import Path
from Annotation import TRAIN_DATA


LABEL = "STORE_NUMBER"

#Setting up pipeline and entity recognizer
nlp = spacy.load('en_core_web_sm')
if 'ner' not in nlp.pipe_names:
  ner = nlp.get_pipe('ner')
  nlp.add_pipe(ner) 
  
#Add new entity label to the recognizer
ner.add_label(LABEL)

#Disable other pipes from training
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
with nlp.disable_pipes(*other_pipes):
  optimizer = nlp.resume_training()
  for itn in range(100):
    random.shuffle(TRAIN_DATA)
    losses = {}
    batches = minibatch(TRAIN_DATA, size = compounding(4.0, 32.0, 1.001))
    for batch in batches:
      texts, annotations = zip(*batch)
      example = []
      for i in range(len(texts)):
        doc = nlp.make_doc(texts[i])
        example.append(Example.from_dict(doc, annotations[i]))
      nlp.update(example, sgd = optimizer, drop = 0.35, losses = losses)
      print("Losses", losses)
      
output_dir = Path("C:/Users/uchei/Desktop")
nlp1.to_disk(output_dir)
print("Saved model to", output_dir)
