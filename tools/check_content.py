"""Validate question integrity and exact notebook provenance, without executing notebooks."""
import json
import runpy
import argparse
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
for item in [*topics,*questions]:
    source=item['source']; path=course/source['path']
    if not course.is_dir():
        assert source['cells'] and all(isinstance(n,int) and n>=1 for n in source['cells'])
        continue
    assert path.is_file(), path
    notebook=json.loads(path.read_text(encoding='utf-8'))
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
    if q['type']=='open': assert len(q['rubric'])>=3
for n in range(1,6):
    assert sum(q['lesson']==n for q in questions)>=20
print('Content checks passed:',len(topics),'topics,',len(questions),'questions')
print('By lesson:',dict(Counter(q['lesson'] for q in questions)))
print('By format:',dict(Counter(q['type'] for q in questions)))
print('Notebook provenance:', 'checked against local files' if course.is_dir() else 'metadata only; pass --course PATH for local notebook checks')
