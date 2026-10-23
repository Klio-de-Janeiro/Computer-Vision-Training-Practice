"""Execute all 65 educational examples on tiny synthetic inputs; requires torch/numpy/scipy/cv2."""
import math
import runpy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import cv2

torch.manual_seed(7)
torch.set_num_threads(2)
data=runpy.run_path(str(Path(__file__).with_name('content.py')))
namespaces={}
for topic in data['TOPICS']:
    if topic.get('supplemental'):continue  # supplementary API cells have their own executable checks
    ns={}
    exec(compile(topic['implementation']['code'],topic['id']+'.py','exec'),ns)
    namespaces[topic['id']]=ns
checked=set()
def get(topic,name):
    checked.add(topic)
    return namespaces[topic][name]
def same(a,b):
    np.testing.assert_allclose(np.asarray(a),np.asarray(b),rtol=1e-5,atol=1e-5)

# Image processing: explicit dtypes, constants, geometry and known answers.
rgb=np.full((32,32,3),100,np.uint8);gray=rgb[...,0].copy()
same(get('raster','bgr_to_gray')(np.array([[[0,0,255]]],np.uint8)),[[76]])
same(get('brightness','adjust')(np.array([250],np.uint8),brightness=20),[255])
glob,clahe=get('histogram','equalize_gray')(gray);assert glob.shape==clahe.shape==gray.shape
same(get('whitebalance','gray_world')(rgb),rgb)
_,otsu,adapt=get('threshold','threshold_variants')(gray);assert set(np.unique(otsu))<={0,255};assert adapt.shape==gray.shape
same(get('blend','alpha_blend')(rgb*2,rgb,.25),np.full_like(rgb,125))
for result in get('blur','smooth_variants')(rgb):same(result,rgb)
mag,angle=get('edges','gradients')(gray);same(mag,np.zeros_like(gray));assert np.isfinite(angle).all()
for result in get('morphology','clean_binary')(np.zeros_like(gray)):assert not result.any()
same(get('fourier','lowpass_fft')(gray),gray)
same(get('geometry','warp')(rgb,np.eye(3),(32,32)),rgb)
template=np.array([[0,50],[100,200]],np.uint8);image=np.zeros((8,8),np.uint8);image[3:5,4:6]=template
position,score=get('matching','locate_template')(image,template);assert position==(4,3) and score>.99
same(get('augmentation','posterize')(np.array([255,127,0],np.uint8),4),[240,112,0])

# Classical geometry: no external models or images.
assert np.isfinite(get('harris','harris_response')(gray)).all()
assert get('sift','sift_matches')(gray,gray)[2]==[]
assert get('orb','orb_correspondences')(gray,gray)[2]==[]
pairs=[[SimpleNamespace(distance=a),SimpleNamespace(distance=b)] for a,b in [(1,5),(4,5),(0,0)]]
assert len(get('ratio','ratio_filter')(pairs))==1
points=np.array([[0,0],[100,0],[100,100],[0,100]],np.float32)
H,inliers=get('ransac','robust_homography')(points,points+[5,7]);same(H,[[1,0,5],[0,1,7],[0,0,1]]);assert inliers.all()
cloned=get('poisson','poisson_insert')(rgb,np.zeros((80,80,3),np.uint8),np.full((32,32),255,np.uint8),(40,40));assert cloned.shape==(80,80,3)
boxes,scores=get('classicaldetectors','hog_people')(np.zeros((128,64,3),np.uint8));assert len(boxes)==len(scores)
K=np.array([[800.,0,320],[0,800,240],[0,0,1]])
board=np.zeros((42,3),np.float32);board[:,:2]=np.mgrid[:7,:6].T.reshape(-1,2)*.04
object_views=[];image_views=[]
for i in range(6):
    pixels,_=cv2.projectPoints(board,np.array([.05*i,.02,-.03*i]),np.array([-.1,.02,1.+.1*i]),K,np.zeros(5))
    object_views.append(board);image_views.append(pixels.astype(np.float32))
rms,Khat,*_=get('camera','calibrate')(object_views,image_views,(640,480));assert rms<.01;np.testing.assert_allclose(Khat,K,rtol=1e-3,atol=.1)
R,t=get('pnp','estimate_pose')(board,image_views[1],K,np.zeros(5));assert np.isfinite(R).all();assert t[2]>0
same(get('kalman','kalman_1d')(10,3,12,1,1),(11.6,.8,.8))

# CNN blocks: parameter counts, shape semantics, train/eval and gradients.
x=torch.randn(2,8,6,10)
dw=get('conv','DepthwiseSeparable')(32,64);assert sum(p.numel() for p in dw.parameters())==2336
assert dw(torch.randn(1,32,8,8)).shape==(1,64,8,8)
dilated,up=get('dilated','spatial_layers')(8);assert dilated(x).shape==x.shape;assert up(x).shape==(2,8,12,20)
y=get('normalization','channel_layer_norm')(x,nn.LayerNorm(8));same(y.detach().mean(1),torch.zeros(2,6,10))
cond=get('adaptive','ConditionalNorm')(8,3);same(cond(x,torch.randn(2,3)).detach(),cond.norm(x).detach())
dp=get('dropout','DropPath')(.25);ones=torch.ones(5000,2,2);assert abs(float(dp(ones).mean())-1)<.04;dp.eval();same(dp(ones),ones)
conv=nn.Conv2d(4,8,3);get('init','initialize_relu_conv')(conv);assert torch.count_nonzero(conv.weight)>0;same(conv.bias.detach(),np.zeros(8))
assert get('resnet','ResidualBlock')(8)(x).shape==x.shape
assert get('mobile','SqueezeExcite')(8)(x).shape==x.shape
assert get('moderncnn','ConvNeXtSketch')(8)(x).shape==x.shape
ce,bce=get('classification','classification_losses')(torch.zeros(2,3),torch.tensor([0,1]),torch.zeros(2,3),torch.zeros(2,3));same(ce,math.log(3));same(bce,math.log(2))
fl=get('focal','binary_focal_loss')(torch.zeros(2),torch.tensor([0,1]),gamma=0);same(fl,.5*math.log(2))
same(get('metrics','precision_recall_f1')(8,2,4),(.8,2/3,8/11))
backbone,head=nn.Linear(3,4),nn.Linear(4,2);before=backbone.weight.detach().clone();optimizer=torch.optim.SGD(head.parameters(),lr=.1)
loss=get('finetune','train_frozen_head')(backbone,head,torch.randn(4,3),torch.tensor([0,1,0,1]),optimizer,F.cross_entropy);assert math.isfinite(loss);same(before,backbone.weight.detach())

# Attention/rotary/cache: mathematical properties, not only output shape.
attn=get('attention','attention');q=torch.randn(1,4,8);k=torch.randn(1,4,8);v=torch.randn(1,4,6)
allow=torch.ones(1,4,4,dtype=torch.bool).tril();y,a=attn(q,k,v,allow);same(a.sum(-1),torch.ones(1,4));assert not a[~allow].any()
assert get('heads','EncoderBlock')()(torch.randn(2,5,64)).shape==(2,5,64)
S=torch.tensor([[0.,0.],[0.,1.]]);assert torch.linalg.matrix_rank(S)==1;assert torch.linalg.matrix_rank(S.softmax(-1))==2
mask=get('masks','causal_allowed')(torch.tensor([3]),3);assert int(mask.sum())==6
same(get('masks','masked_token_loss')(torch.zeros(1,3,2),torch.tensor([[0,-100,1]])),math.log(2))
moe=get('moe','TinyMoE')();z=moe(torch.randn(5,32));z.square().sum().backward();assert z.shape==(5,32);assert moe.router.weight.grad is not None
pe=get('position','sinusoidal_pe')(5,8);same(pe[0,0::2],np.zeros(4));same(pe[0,1::2],np.ones(4))
rotate=get('rope','rope');u=torch.randn(2,5,8);pos=torch.arange(5);rot=rotate(u,pos);same(rot.norm(dim=-1),u.norm(dim=-1))
shifted=rotate(q,torch.arange(4)+10) @ rotate(k,torch.arange(4)+10).transpose(-2,-1)
original=rotate(q,torch.arange(4)) @ rotate(k,torch.arange(4)).transpose(-2,-1);same(shifted,original)
coords=torch.stack([torch.zeros(5),torch.arange(5),torch.arange(5)],-1)
multi=get('mrope','mrope')(torch.randn(1,5,12),coords,(4,4,4),rotate);assert multi.shape==(1,5,12)
same(get('bert','mlm_loss')(torch.zeros(1,3,2),torch.tensor([[0,1,0]]),torch.tensor([[False,True,False]])),math.log(2))
same(get('training','clm_loss')(torch.zeros(1,4,2),torch.tensor([[0,1,0,1]]),torch.tensor([[0,0,1,1]])),math.log(2))
cached=get('cache','cached_step');cache=None
for i in range(4):
    result,cache=cached(q[:,None,i:i+1],k[:,None,i:i+1],v[:,None,i:i+1],cache)
    full,_=attn(q[:,i:i+1],k[:,:i+1],v[:,:i+1]);same(result[:,0],full)
assert get('vit','PatchEmbedding')()(torch.randn(1,3,32,48)).shape==(1,6,64)
partition=get('swin','window_partition');assert partition(torch.randn(1,4,4,8),2).shape==(4,4,8)
assert namespaces['swin']['PatchMerging'](8)(torch.randn(1,4,4,8)).shape==(1,2,2,16)
features=torch.eye(2);clip=get('clip','clip_loss')(features,features,torch.tensor(0.));same(clip,math.log1p(math.exp(-1)))
same(get('coca','coca_objective')(torch.tensor(.5),torch.zeros(1,2,2),torch.tensor([[0,1]])),.5+2*math.log(2))
same(get('siglip','siglip_loss')(torch.zeros(2,3),torch.zeros(2,3),torch.tensor(0.),torch.tensor(0.)),2*math.log(2))
packed=get('navit','packed_image_mask')(torch.tensor([0,0,1,1,1]));assert int(packed.sum())==13
same(namespaces['navit']['per_image_mean'](torch.tensor([[1.],[3.],[9.]]),torch.tensor([0,0,1]),2),[[2],[9]])
assert get('naflex','exact_two_to_one_grid')(1000)=={'height':352,'width':704,'tokens':968}

# Detection conventions, assignments and synthetic decoding.
same(get('boxes','xyxy_to_cxcywh')([[0,0,10,20]]),[[5,10,10,20]])
iou=get('iou','iou_one_to_many');same(iou([0,0,10,10],[[5,0,15,10]]),[1/3])
same(get('map','ap_all_points')([.5,1.],[1.,.5]),.75)
assert get('nms','nms_single_class')([[0,0,10,10],[0,0,10,10]],[.9,.8],iou)==[0]
same(get('anchors','decode_offsets')(torch.tensor([[5.,5.,10.,10.]]),torch.zeros(1,4)),[[0,0,10,10]])
roi=get('rcnn','roi_pool_sketch')(torch.ones(1,2,8,8),(0,0,4,4),(2,2));same(roi,torch.ones(1,2,2,2))
same(get('ssd','ssd_classification_mining')(torch.tensor([1.,2.,3.,4.,5.]),torch.tensor([True,False,False,False,False])),13)
same(get('yolo','decode_yolov3')(torch.zeros(1,4),torch.zeros(1,2),torch.tensor([[10.,20.]]),8),[[-1,-6,9,14]])
fpn=get('retinanet','TinyFPN')();levels=fpn([torch.randn(1,32,17,17),torch.randn(1,64,9,9),torch.randn(1,128,5,5)]);assert [tuple(y.shape) for y in levels]==[(1,32,17,17),(1,32,9,9),(1,32,5,5)]
heat=torch.zeros(1,1,5,5);heat[0,0,2,2]=.9;peaks=get('centernet','center_peaks')(heat);assert int(peaks.sum())==1
same(get('fcos','centerness_target')(torch.tensor([[4.,3.,4.,3.],[0.,3.,4.,3.]])),[1,0])
cost=torch.tensor([[1.,1.1],[2.,100.]]);qi,gi=get('detr','match_slots')(cost);same(cost[qi,gi].sum(),3.1)
assert checked==set(namespaces),(set(namespaces)-checked)
print(f'Executed and verified {len(checked)} examples on CPU: shape, arithmetic, masks, gradients, geometry, assignment and train/eval checks passed.')
