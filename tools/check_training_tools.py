"""Execute the exact published cells in temporary directories; no datasets or remote services."""
import ast
import os
import runpy
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING'] = 'false'
os.environ['MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR'] = 'false'
os.environ['MLFLOW_TRACKING_URI'] = 'sqlite:///training.db'
os.environ['MLFLOW_REGISTRY_URI'] = 'sqlite:///training.db'
os.environ['DO_NOT_TRACK'] = 'true'

data=runpy.run_path(str(Path(__file__).with_name('content.py')))
cards=[c for c in data['CODE_CARDS'] if c['runtime']=='tooling']
assert len(cards)>=40
for card in cards:
    with tempfile.TemporaryDirectory(prefix='cv-tool-') as directory:
        # A fresh process also isolates MLflow's cached tracking stores and writer threads.
        source=card['setup']+'\n'+card['code']+'\n'+card['test']
        ast.parse(source,filename=card['id'])
        result=subprocess.run([sys.executable,'-'],input='import torch\ntorch.set_num_threads(2)\ntorch.manual_seed(7)\n'+source,
                              cwd=directory,capture_output=True,text=True,encoding='utf-8',timeout=120)
        if result.returncode:
            raise AssertionError(f"Published cell {card['id']}:\n{result.stdout}\n{result.stderr}")
        print('OK',card['id'],flush=True)
print('Training tool cells passed:',len(cards),dict(Counter(c['library'] for c in cards)))
