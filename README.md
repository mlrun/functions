# function hub (wip)


This functions hub is intended to be a centralized location for open source contributions of function components.  These are functions expected to be run as independent mlrun pipeline compnents, and as public contributions, it is expected that contributors follow certain guidelines/protocols (please chip-in).

Currently a list of functions is maintained in the file **[functions-table](functions.table.csv)**, and a script is run over each entry.  This should disappear in the next version.  All information regarding how a function is built, etc..., should be stored with that function, including its tests.  This metadata could be a yaml file which in the end gets filled with the function source code

each function bundle has<br>
* one development file (ipynb, py)<br>
* one final code file (ipynb, py)<br>
* one corresponding yaml file, although this is likely to be auto-generated<br>
* one test file (ipynb)<br>
    - preferrably a clean and documented ipynb that can be run under CI<br>
* one readme/tutorial (ipynb)


### code

#### format style of function
#### commenting
#### documentation
#### code revision

### yaml (autogen)

### testing
