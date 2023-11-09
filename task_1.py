from transformers import pipeline
import pprint 

unmasker = pipeline('fill-mask', model='xlm-roberta-base')

encoded_input = unmasker("Hello I'm a <mask> man.")

pp = pprint.PrettyPrinter(indent=4)
pp.pprint(encoded_input)


