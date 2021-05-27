from bert_embeddings import init_context,handler
import nuclio
import json
import pickle
import mlrun
import subprocess
import re
from pygit2 import Repository

ARCHIVE = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
ARTIFACTS_PATH = 'artifacts'

def _set_mlrun_hub_url(repo_name = None, branch_name = None, function_name = None):
    repo_name =  re.search("\.com/.*?/", str(subprocess.run(['git', 'remote', '-v'], stdout=subprocess.PIPE).stdout)).group()[5:-1] if not repo_name else repo_name
    branch_name = Repository('.').head.shorthand if not branch_name else branch_name
    function_name = "" if not function_name else function_name # MUST ENTER FUNCTION NAME !!!!
    hub_url = f"https://raw.githubusercontent.com/{repo_name}/functions/{branch_name}/{function_name}/function.yaml"
    mlrun.mlconf.hub_url = hub_url

def test_run_local_bert_embeddings():
    event = nuclio.Event(body=json.dumps(['John loves Mary']))
    ctx = nuclio.Context()
    init_context(ctx)
    outputs = pickle.loads(handler(ctx, event))
    print(outputs)

def test_run_imported_bert_embeddings():
    event = nuclio.Event(body=json.dumps(['John loves Mary']))
    ctx = nuclio.Context()
    _set_mlrun_hub_url(function_name="bert_embeddings")
    fn = mlrun.import_function("hub://bert_embeddings")
    outputs = fn.run(params={"context": ctx,
                             "event": event})
    print(outputs)
