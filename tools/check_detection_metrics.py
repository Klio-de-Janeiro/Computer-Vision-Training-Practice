"""Check the published AP/IoU worked example, then compare against official COCOeval."""
import argparse
import contextlib
import io
import runpy
from pathlib import Path
import numpy as np

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--core-only',action='store_true',help='Skip the pycocotools comparison when the package is unavailable')
args=parser.parse_args()
data=runpy.run_path(str(Path(__file__).with_name('content.py')))
topics={t['id']:t for t in data['TOPICS']}
ns={};exec(compile(topics['map']['implementation']['code'],'published-ap','exec'),ns)
ap=ns['ap_all_points']
assert abs(ns['ap']-5/6)<1e-12
assert abs(ap([0,.5,1],[0,.5,2/3])-2/3)<1e-12  # FP, TP, TP
assert abs(ap([.5,1,1],[1,1,2/3])-1)<1e-12    # TP, TP, FP
assert ap([0,0],[0,0])==0
assert abs(ap([.5,.5,.5,.5],[1,.5,1/3,.25])-.5)<1e-12
box_ns={};exec(compile(topics['iou']['implementation']['code'],'published-iou','exec'),box_ns)
iou=box_ns['iou_one_to_many']
np.testing.assert_allclose(iou([0,0,10,10],[[5,0,15,10]]),[1/3])
np.testing.assert_allclose(iou([20,0,30,10],[[20,0,26,10]]),[.6])
np.testing.assert_allclose(iou([0,0,10,10],[[10,0,20,10],[0,0,10,10]]),[0,1])
matrix=np.array([[1.,0.],[1.,0.],[0.,.6],[0.,0.]])
def matching(threshold):
    occupied=set();flags=[]
    for row in matrix:
        candidates=[j for j,value in enumerate(row) if j not in occupied and value>=threshold]
        if candidates:occupied.add(max(candidates,key=lambda j:row[j]));flags.append(1)
        else:flags.append(0)
    return np.array(flags)
def pr(flags):
    tp=flags.cumsum();return tp/2,tp/np.arange(1,len(tp)+1)
def ap101(recall,precision):
    return np.mean([max((p for r,p in zip(recall,precision) if r>=threshold),default=0.) for threshold in np.linspace(0,1,101)])
np.testing.assert_array_equal(matching(.5),[1,0,1,0])
np.testing.assert_array_equal(matching(.75),[1,0,0,0])
r50,p50=pr(matching(.5));r75,p75=pr(matching(.75))
assert abs(ap(r50,p50)-5/6)<1e-12 and abs(ap(r75,p75)-.5)<1e-12
assert abs(ap101(r50,p50)-253/303)<1e-12 and abs(ap101(r75,p75)-51/101)<1e-12
assert abs(np.mean([ap101(*pr(matching(t))) for t in np.linspace(.5,.95,10)])-61/101)<1e-12
assert abs(p50.mean()-2/3)<1e-12
print('Detection example passed: geometry, duplicates, confidence prefixes, ranking, all-points AP and COCO101 arithmetic.')
if args.core_only:raise SystemExit(0)

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
with contextlib.redirect_stdout(io.StringIO()):
    gt=COCO()
    gt.dataset=dict(info={},images=[dict(id=1,width=50,height=10)],
                    categories=[dict(id=1,name='cat'),dict(id=2,name='absent')],
                    annotations=[dict(id=1,image_id=1,category_id=1,bbox=[0,0,10,10],area=100,iscrowd=0),
                                 dict(id=2,image_id=1,category_id=1,bbox=[20,0,10,10],area=100,iscrowd=0)])
    gt.createIndex()
    pred=gt.loadRes([dict(image_id=1,category_id=1,bbox=box,score=score) for box,score in
                    [([0,0,10,10],.95),([0,0,10,10],.90),([20,0,6,10],.80),([40,0,10,10],.70)]])
    evaluator=COCOeval(gt,pred,'bbox')
    evaluator.evaluate();evaluator.accumulate()
official=evaluator.eval['precision'][:,:,0,0,-1]  # [IoU, recall], cat, area=all, maxDets=100
i75=np.flatnonzero(np.isclose(evaluator.params.iouThrs,.75))[0]
assert abs(official[0].mean()-253/303)<1e-12
assert abs(official[i75].mean()-51/101)<1e-12
assert abs(official.mean()-61/101)<1e-12
assert np.all(evaluator.eval['precision'][:,:,1,0,-1]==-1)  # class without GT is undefined
first=evaluator.evalImgs[0]
np.testing.assert_array_equal(first['dtMatches'][0]>0,[True,False,True,False])
np.testing.assert_array_equal(first['dtMatches'][i75]>0,[True,False,False,False])
print('Official COCOeval agrees: AP50=253/303, AP75=51/101, AP@[.50:.95]=61/101; missing-GT class excluded.')
