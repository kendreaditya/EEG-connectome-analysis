import sys
import importlib

args = sys.argv
print(args)
model = importlib.import_module(args[1])

print(model.vgg11)
