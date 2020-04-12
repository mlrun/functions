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

        entry = {'description': fn.spec.description,
                 'categories': fn.metadata.categories,
                 'kind': fn.kind,
                 'versions': {'latest': f'hub://{funcdir}'}}

        print(fn.metadata.name, entry)
        catalog[fn.metadata.name] = entry


with open('catalog.yaml', 'w') as fp:
    fp.write(yaml.dump(catalog))

with open('catalog.json', 'w') as fp:
    fp.write(json.dumps(catalog))
