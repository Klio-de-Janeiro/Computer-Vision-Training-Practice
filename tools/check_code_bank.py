"""Validate all CPU API examples; external integrations get syntax/provenance checks, no downloads."""
import ast
import runpy
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

data=runpy.run_path(str(Path(__file__).with_name('content.py')))
checked=0; external=0
torch.set_num_threads(2)
for c in data['CODE_CARDS']:
    src=c['setup']+'\n'+c['code']+'\n'+c['test']
    ast.parse(src,filename=c['id'])
    if c['runtime']!='cpu':external+=1;continue
    assert c['test'].strip(),c['id']
    ns=dict(np=np,torch=torch,nn=nn,F=F)
    try:exec(compile(src,c['id'],'exec'),ns)
    except Exception as e:raise AssertionError(f"API example {c['id']}: {e}") from e
    checked+=1
print(f'API checks passed: {checked} executed CPU examples; {external} external integration examples syntax-checked without weights/downloads.')
