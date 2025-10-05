## Available Commands
(Explore more advanced options in the code, this is basic usage demonstration)

### generate-item-yaml
Generate an `item.yaml` file (basic draft) in the appropriate directory from a Jinja2 template

Usage:
    `python -m cli.cli generate-item-yaml TYPE NAME`

Example:
    `python -m cli.cli generate-item-yaml function aggregate`

---

### item-to-function
Creates a `function.yaml` file based on a provided `item.yaml` file.

Usage:
    `python -m cli.cli item-to-function --item-path PATH`

Example:
    `python -m cli.cli item-to-function --item-path functions/src/aggregate`

---

### function-to-item
Creates a `item.yaml` file based on a provided `function.yaml` file.

Usage:
    `python -m cli.cli function-to-item PATH`

Example:
    `python -m cli.cli function-to-item --path functions/src/aggregate`

---

### run-tests
Run assets test suite.

Usage:
    `python -m cli.cli run-tests -r PATH -s TYPE -fn NAME`

Example:
    `python -m cli.cli run-tests -r functions/src/aggregate -s py -fn aggregate`

---

### build-marketplace
Build and push (create a PR) the updated marketplace/<TYPE> directory (e.g: marketplace/functions)

Usage:
    `python -m cli.cli build-marketplace -s SOURCE-DIR -sn TYPE -m MARKETPLACE-DIR -c CHANNEL -v -f`

Example:
    `python -m cli.cli build-marketplace -s ./functions/src -sn functions -m marketplace -c master -v -f`

---

### update-readme
Regenerate the `README.md` files in each of the asset directories (functions/modules).

Usage:
    `python -m cli.cli update-readme --asset TYPE`

Example:
    `python -m cli.cli update-readme --asset functions --asset modules`