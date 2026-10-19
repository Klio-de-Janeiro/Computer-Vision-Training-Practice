"""Validate question integrity and exact notebook provenance, without executing notebooks."""
import json
import runpy
import argparse
import ast
from collections import Counter
from pathlib import Path

app=Path(__file__).resolve().parents[1]
data=runpy.run_path(str(app/'tools/content.py'))
topics=data['TOPICS']; questions=data['QUESTIONS']
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--course',type=Path,default=app.parents[1]/'cv_course_at_mipt')
args=parser.parse_args()
course=args.course
assert len({t['id'] for t in topics})==len(topics)
assert len({q['id'] for q in questions})==len(questions)
ids={t['id'] for t in topics}
notebooks={}
implementations=[{'id':'code-'+t['id'],'source':t['implementation']['source']} for t in topics]
diagrams=data['DIAGRAMS']
for item in [*topics,*questions,*implementations,*diagrams]:
    source=item['source']; path=course/source['path']
    if not course.is_dir():
        assert source['cells'] and all(isinstance(n,int) and n>=1 for n in source['cells'])
        continue
    assert path.is_file(), path
    if path not in notebooks:
        notebooks[path]=json.loads(path.read_text(encoding='utf-8'))
    notebook=notebooks[path]
    for n in source['cells']:
        assert 1<=n<=len(notebook['cells']), (item['id'],n)
        assert ''.join(notebook['cells'][n-1]['source']).strip(),(item['id'],n)
for q in questions:
    assert q['topic'] in ids
    assert q['answer'].strip() and q['example'].strip()
    if q['type'] in ('single','multi'):
        assert len(q['options'])==len(set(q['options']))
        assert all(0<=i<len(q['options']) for i in q['correct'])
        assert len(set(q['correct']))==len(q['correct'])
        assert len(q['correct'])==1 if q['type']=='single' else 1<len(q['correct'])<len(q['options'])
    if q['type']=='short': assert q['aliases']
    if q['type']=='number': assert q['tolerance']>=0
    if q['type']=='number': assert q['calculation'].strip()
    if q['type']=='open': assert len(q['rubric'])>=3
for n in range(1,6):
    assert sum(q['lesson']==n for q in questions)>=20
print('Content checks passed:',len(topics),'topics,',len(questions),'questions')
for t in topics:
    assert t['mathNotes'].strip() and '$$' in t['mathNotes'], t['id']
    assert t['implementation']['note'].strip(),t['id']
    ast.parse(t['implementation']['code'],filename=t['id']+'.py')
print('Mathematical notes and Python syntax checked:',len(topics),'examples; worked LaTeX:',sum(q['type']=='number' for q in questions))
assert len(diagrams)==40 and len({d['id'] for d in diagrams})==len(diagrams)
for d in diagrams:
    assert d['topic'] in ids and d['summary'].strip()
    nodes={n['id']:n for n in d['nodes']}
    assert len(nodes)==len(d['nodes'])
    assert len({(n['col'],n['row']) for n in d['nodes']})==len(nodes)
    assert all(n['title'] and n['hint'] and n['detail'] for n in nodes.values())
    assert all(n['kind'] in ['block','input','output','loss'] for n in nodes.values())
    assert all(isinstance(n['col'],int) and isinstance(n['row'],int) and n['col']>=0 and n['row']>=0 for n in nodes.values())
    assert all(e['from'] in nodes and e['to'] in nodes and e['from']!=e['to'] for e in d['edges'])
    assert all(e['kind'] in ['flow','skip','context','training'] for e in d['edges'])
    incoming={key:sum(e['to']==key for e in d['edges']) for key in nodes}
    queue=[key for key,n in incoming.items() if n==0];visited=[]
    while queue:
        key=queue.pop();visited.append(key)
        for edge in d['edges']:
            if edge['from']==key:
                incoming[edge['to']]-=1
                if incoming[edge['to']]==0:queue.append(edge['to'])
    assert len(visited)==len(nodes),d['id']
for t in topics:assert t['diagrams']==[d['id'] for d in diagrams if d['topic']==t['id']]
print('Architecture diagrams checked:',len(diagrams),'graphs;',sum(len(d['nodes']) for d in diagrams),'blocks')
print('By lesson:',dict(Counter(q['lesson'] for q in questions)))
print('By format:',dict(Counter(q['type'] for q in questions)))
architectures=data.get('ARCHITECTURES',[])
counts=Counter(q.get('architecture') for q in questions)
assert len(architectures)==11
assert all(counts[a['id']]>=30 for a in architectures)
assert len({q['prompt'].strip().casefold() for q in questions})==len(questions),'Repeated question wording'
print('Architecture coverage:',{a['title']:counts[a['id']] for a in architectures})
print('Notebook provenance:', 'checked against local files' if course.is_dir() else 'metadata only; pass --course PATH for local notebook checks')
