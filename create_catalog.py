from mlrun import import_function
import yaml
import json
from pathlib import Path
import os

catalog = {}

for f in Path('.').glob('**/*.yaml'):
    specpath = str(f).replace('\\', '/')
    print('path:', specpath)

    if os.path.isfile(specpath):
        try:
            fn = import_function(specpath)
        except Exception as e:
            print(f'failed to load func {specpath} , {e}')
            continue

        if not fn.kind or fn.kind in ['', 'local', 'handler']:
            print(f'illegal function or kind in {specpath}')
            continue

        if fn.metadata.name in catalog:
            entry = catalog[fn.metadata.name]
        else:
            entry = {'description': fn.spec.description,
                     'categories': fn.metadata.categories,
                     'kind': fn.kind,
                     'versions': {}}

        entry['versions'][fn.metadata.tag or 'latest'] = specpath
        print(fn.metadata.name, entry)
        catalog[fn.metadata.name] = entry


with open('catalog.yaml', 'w') as fp:
    fp.write(yaml.dump(catalog))

with open('catalog.json', 'w') as fp:
    fp.write(json.dumps(catalog))