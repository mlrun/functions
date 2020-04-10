from mlrun import import_function
import yaml
import json

import os

catalog = {}
dirs = os.listdir('.')
for funcdir in dirs:
    specpath = os.path.join(funcdir, 'function.yaml')
    if os.path.isfile(specpath):
        fn = import_function(specpath)

        entry = {'name': fn.metadata.name,
                 'description': fn.spec.description,
                 'categories': fn.metadata.categories,
                 'versions': {'latest': f'hub://{funcdir}'}}

        print(entry)
        catalog[fn.metadata.name] = entry


with open('catalog.yaml', 'w') as fp:
    fp.write(yaml.dump(catalog))

with open('catalog.json', 'w') as fp:
    fp.write(json.dumps(catalog))
