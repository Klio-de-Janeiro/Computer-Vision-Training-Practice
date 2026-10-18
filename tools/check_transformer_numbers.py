"""Independently recompute the authored Transformer calculation answers."""
import math
import runpy
from pathlib import Path

data = runpy.run_path(str(Path(__file__).with_name('content.py')))
questions = {q['id']: q for q in data['QUESTIONS']}
expected = {}
def bank(name, **answers):
    expected.update({f'tx-{name}-{key.replace("_", "-")}': value for key, value in answers.items()})

bank('bert', selected_count=2000*.15, mask_count=300*.8,
     random_share=100*.15*.1, score_elements=128**2, head_dim=768/12)
bank('vit', patch_count=(256//16)*(384//16), patch_vector=8*8*3,
     patch_params=16*16*3*256, position_params=(196+1)*768, pair_count=(64+1)**2)
bank('swin', window_tokens=7**2, window_score=7**4, window_count=(56//7)**2,
     total_pairs=(56//7)**2*7**4, relative_table=(2*7-1)**2,
     relative_heads=6*(2*7-1)**2, merged_count=(56//2)**2, shift_units=7//2)
bank('clip', pair_matrix=8**2, negatives=8*(8-1), projection=768*256,
     cosine=(3*0+4*5)/(math.hypot(3,4)*math.hypot(0,5)), symmetric_loss=(.4+.6)/2)
bank('coca', weighted_loss=1*.5+2*1.5, cross_pairs=20*196, two_heads=8*20*196,
     projection_head=768*512, unimodal_causal=sum(range(1,11)))
bank('siglip', positive_fraction=20/20**2, negative_pairs=16**2-16,
     scaled_logit=10*.3-2, threshold_cos=6/10, block_count=(1000//250)**2)
bank('navit', rect_patches=(224//16)*(448//16), two_lengths=196+392,
     fill_padding=1024-sum([392,392,196]), ordinary_pad=3*400-sum([100,200,400]),
     block_pairs=4**2+3**2, forbidden_cross=2*4*3, max_fitting=1000//196,
     area_scaling=3**2, block_ratio=200**2/(100**2+100**2), normalized_use=800/1000)
bank('naflex', budget_square=math.isqrt(1024)*16,
     ratio_two=max(h for h in range(1,513) if h*(2*h)<=512), ratio_two_width=32*16,
     unused_cap=1000-22*(2*22), ceil_example=math.ceil(31/16)*math.ceil(47/16),
     embedding_projection=768*1152, fill_batch=512-384, attention_factor=1024**2/256**2,
     max_width_fixed_height=600//20, area_from_tokens=300*8**2)
bank('moe', expert_total=8*2, active_params=2*3, relative_all=16/2, assignments=1000*2,
     balanced_load=2000/8, router_weights=256*8, expert_ffn=2*128*512,
     mixture_output=.75*2+.25*6, topk_factor=4/1, active_fraction=2/32)
bank('detr', empty_slots=100-7, cross_count=100*196, query_self_count=100**2,
     embedding_params=100*256, hungarian_toy=min(1+100,2+1.1))
bank('transformer', scale_value=math.sqrt(64), allowed_triangle=sum(range(1,9)),
     pe_size=512*768, score_size=12*20, ffn_size=2*64*256)

actual = {key for key,q in questions.items() if key.startswith('tx-') and q['type']=='number'}
assert actual==set(expected), (actual-set(expected),set(expected)-actual)
for key,value in expected.items():
    assert math.isclose(questions[key]['value'],value,rel_tol=1e-12,abs_tol=1e-12),(key,value)
print(f'Independently verified {len(expected)} new numerical answers')
