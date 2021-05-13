from mlrun import import_function
import yaml
import json
from pathlib import Path
import os

catalog = {}

file_list = Path('.').glob('**/*.yaml')

sorted_file_list = sorted(file_list, key=lambda f: str(f))

for f in sorted_file_list:
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
            dirpath = os.path.dirname(specpath)
            docfiles = [nb for nb in os.listdir(dirpath) if nb.endswith('.ipynb')]
            docfile = dirpath if not docfiles else os.path.join(dirpath, docfiles[0]).replace('\\', '/')
            entry = {'description': fn.spec.description,
                     'categories': fn.metadata.categories,
                     'kind': fn.kind,
                     'docfile': docfile,
                     'versions': {}}

        entry['versions'][fn.metadata.tag or 'latest'] = specpath
        print(fn.metadata.name, entry)
        catalog[fn.metadata.name] = entry


with open('catalog.yaml', 'w') as fp:
    fp.write(yaml.dump(catalog))

with open('catalog.json', 'w') as fp:
    fp.write(json.dumps(catalog))

mdheader = '''# Functions hub 

This functions hub is intended to be a centralized location for open source contributions of function components.  
These are functions expected to be run as independent mlrun pipeline compnents, and as public contributions, 
it is expected that contributors follow certain guidelines/protocols (please chip-in).

## Functions

'''


def gen_md_table(header, rows=None):
    rows = [] if rows is None else rows

    def gen_list(items=None):
        items = [] if items is None else items
        out = '|'
        for i in items:
            out += ' {} |'.format(i)
        return out

    out = gen_list(header) + '\n' + gen_list(len(header) * ['---']) + '\n'
    for r in rows:
        out += gen_list(r) + '\n'
    return out


with open('README.md', 'w') as fp:
    fp.write(mdheader)
    rows = []
    for k, v in catalog.items():
        kind = v['kind']
        if kind == 'remote':
            kind = 'nuclio'
        rows.append([f"[{k}]({v['docfile']})", kind,
                    v['description'], ', '.join(v['categories'] or [])])

    text = gen_md_table(['function', 'kind', 'description', 'categories'], rows)
    fp.write(text)
