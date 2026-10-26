"""CPU-only architecture illustrations using locally matched real DICOM cache rows."""
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/local_global/architecture_illustration'
TARGETS=['ACL','MCL','Medial meniscus','Lateral meniscus','Medial OA','Lateral OA',
         'PF OA','Effusion','Synovitis',"Baker’s",'Contusion','Fracture']

def encode(array):
    b=io.BytesIO();Image.fromarray(array).save(b,format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()

def sample():
    if not (OUT/'local_sample_records.json').exists():
        old={}
        for p in (ROOT/'data/external/topk-support-cache-shards').glob('*/manifest.jsonl'):
            for line in p.read_text().splitlines():
                r=json.loads(line);r['directory']=str(p.parent.relative_to(ROOT));old[(r['study_id'],r['series_id'])]=r
        train={r['study_id'] for r in csv.DictReader((ROOT/'data/local_global/assets/folds.csv').open()) if r['fold']!='0'}
        groups={}
        native=ROOT/'data/local_global/shard_0_audit/native_triplets/shard_0/slice_cache/manifest.jsonl'
        for line in native.read_text().splitlines():
            r=json.loads(line);b=old.get((r['study_id'],r['series_id']))
            if not b or r['study_id'] not in train:continue
            available=[i for i,offsets in enumerate(r['triplet_offsets']) if all(r['files'][j] in b['files'] for j in offsets)]
            if not available:continue
            i=min(available,key=lambda k:abs(k-(r['count']-1)/2))
            files=[r['files'][j] for j in r['triplet_offsets'][i]]
            r.update(example_window=i,local_directory=b['directory'],local_part=b['part'],local_support=b['support_part'],
                     local_rows=[b['row_start']+b['files'].index(f) for f in files])
            groups.setdefault(r['study_id'],[]).append(r)
        matched=next(sorted(rs,key=lambda r:r['slot']) for rs in groups.values()
                     if len(rs)==4 and {r['plane'].lower() for r in rs}=={'sagittal','coronal','axial'})
        (OUT/'local_sample_records.json').write_text(json.dumps(matched,indent=2)+'\n')
    rows=json.loads((OUT/'local_sample_records.json').read_text())
    result=[];provenance=[]
    for r in rows:
        directory=ROOT/r['local_directory']
        images=np.load(directory/r['local_part'],mmap_mode='r')[r['local_rows']].copy()
        support=np.load(directory/r['local_support'],mmap_mode='r')[r['local_rows']].copy()
        i=r['example_window'];offsets=r['triplet_offsets'][i]
        indices=[r['native_indices'][j] for j in offsets]
        assert indices==[indices[1]-1,indices[1],indices[1]+1]
        result.append({'slot':r['slot'],'plane':r['plane'], 'fluid':r['fluid_sensitive'],
                       'fat':r['fat_suppression'], 'indices':indices, 'position':round(r['positions'][i],3),
                       'gaps':[round(v,2) for v in r['neighbor_gaps_mm'][i]],
                       'images':[encode(a) for a in images],'support':[encode(a) for a in support], 'pixels':images})
        provenance.append({'study_id':r['study_id'],'series_id':r['series_id'],'slot':r['slot'],
                           'native_indices':indices,'dicom_files':[r['files'][j] for j in offsets],
                           'local_part':str(directory/r['local_part']),'local_rows':r['local_rows'],
                           'pixel_sha256':hashlib.sha256(images.tobytes()).hexdigest(),
                           'source':'Local original 140mm/224px centre cache; exact native filenames matched to new manifest'})
    (OUT/'sample_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    return result

BLUE='#dcebf9';GREEN='#e3f1e6';ORANGE='#ffecd5';PURPLE='#ece4f8';INK='#182a3b'
def box(ax,x,y,w,h,title,detail='',color=BLUE,size=15):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008,rounding_size=0.012',
                              facecolor=color,edgecolor='#a4b1ba',linewidth=1.1))
    ax.text(x+w/2,y+h*.66,title,ha='center',va='center',fontsize=size,color=INK,weight='bold')
    if detail:ax.text(x+w/2,y+h*.28,detail,ha='center',va='center',fontsize=12,color=INK,linespacing=1.35)
def arrow(ax,a,b,color='#52677b',style='-',label=None):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=15,lw=1.6,color=color,linestyle=style))
    if label:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.016,label,ha='center',fontsize=11,color=color)

def overview(samples):
    fig=plt.figure(figsize=(18,12),facecolor='white');ax=fig.add_axes([.03,.04,.94,.91]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    fig.text(.03,.975,'Local + global MRI reader | one jointly trained ResNet50 model',fontsize=23,color=INK,weight='bold',va='top')
    ax.text(.02,.925,'REAL DICOM-DERIVED INPUT',fontsize=14,color=INK,weight='bold')
    for row,s in enumerate(samples):
        y=.74-row*.172
        ax.text(.01,y+.14,f"Slot {s['slot']} · {s['plane']} · fluid={s['fluid']} · FS={s['fat']}",fontsize=11,color=INK)
        for k,a in enumerate(s['pixels']):
            axy=fig.add_axes([.046+k*.070,.04+.91*y,.064,.118]);axy.imshow(a,cmap='gray',vmin=0,vmax=255);axy.axis('off')
            axy.set_title(['Previous','Centre','Next'][k],fontsize=10,color=INK,pad=3)
    ax.text(.02,.17,'3 acquired neighbours + 3 support masks\n140 mm crop · 224 × 224 · 6 channels/window',fontsize=12,color=INK)
    box(ax,.295,.665,.235,.17,'Shared ResNet50','ImageNet initialization; fully trainable\nOne 2,048-D vector per window',BLUE)
    arrow(ax,(.245,.735),(.289,.75))
    ax.text(.315,.61,'Train: random 24 of up to 48 windows\nEvaluation: all available windows (up to 48)',fontsize=12,color=INK)
    box(ax,.585,.795,.225,.135,'Local prediction path','12 logits per window\nTop 10% mean, separately per finding',GREEN)
    arrow(ax,(.535,.79),(.578,.85))
    box(ax,.585,.595,.225,.135,'Project + metadata','2,048 → 256\nPlane, fluid, fat, slot, position, gaps',ORANGE)
    arrow(ax,(.535,.705),(.578,.66))
    box(ax,.585,.39,.225,.135,'Small Transformer','1 encoder layer · 4 heads\n24/48 window tokens × 256 features',ORANGE)
    arrow(ax,(.697,.59),(.697,.533))
    box(ax,.585,.205,.225,.135,'Finding-specific pooling','12 learned queries attend to tokens\n12 corrections × fixed scale 0.1',ORANGE)
    arrow(ax,(.697,.385),(.697,.347))
    box(ax,.867,.57,.108,.135,'Add + sigmoid','12 probabilities',PURPLE,size=13)
    arrow(ax,(.817,.86),(.92,.713))
    arrow(ax,(.817,.27),(.92,.566))
    box(ax,.30,.01,.42,.115,'Weighted soft-label BCE','HPO v3.1-D2 targets → AdamW → all trainable paths',PURPLE,size=15)
    arrow(ax,(.92,.566),(.724,.075),color='#91602c',style='--')
    arrow(ax,(.505,.137),(.41,.657),color='#91602c',style='--')
    ax.text(.815,.04,'Dashed arrows: training gradients\nNo teacher, frozen encoder or decoder',fontsize=11,color=INK,ha='center')
    fig.text(.03,.015,'Real acquired slice samples from one report-training knee. Feature vectors and connections are schematic; no attention weights or clinical findings are claimed.',fontsize=11,color=INK)
    for ext in ('png','svg'):fig.savefig(OUT/f'local_global_architecture.{ext}',dpi=160,facecolor='white')
    plt.close(fig)

def transformer():
    fig=plt.figure(figsize=(17,7),facecolor='white');ax=fig.add_axes([.035,.06,.93,.83]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    fig.text(.035,.965,'Inside the small Transformer: windows exchange information within one knee',fontsize=22,color=INK,weight='bold',va='top')
    box(ax,.01,.61,.19,.18,'Window tokens','24/48 × 256\nResNet features + metadata',BLUE)
    box(ax,.27,.61,.215,.18,'4-head self-attention','Q, K, V from the same tokens\n64 features per head',ORANGE)
    box(ax,.56,.61,.20,.18,'Feed-forward update','256 → 512 → 256\nReLU; residuals + LayerNorm',ORANGE)
    box(ax,.83,.61,.16,.18,'Context tokens','24/48 × 256',GREEN)
    for a,b in [((.204,.7),(.26,.7)),((.49,.7),(.55,.7)),((.765,.7),(.82,.7))]:arrow(ax,a,b)
    ax.text(.505,.91,'ONE encoder layer · dropout 0.1 · padded windows masked · no mixing between knees',ha='center',fontsize=13,color=INK)
    box(ax,.02,.17,.22,.18,'12 learned queries','One learned vector per finding\nACL, MCL, meniscus, OA, …',PURPLE)
    box(ax,.34,.17,.26,.18,'Query cross-attention','Queries retrieve from context tokens\n4 heads → 12 × 256',ORANGE)
    box(ax,.71,.17,.27,.18,'12 logit corrections','LayerNorm → shared Linear(256, 1)\nMultiply by 0.1; add to local logits',GREEN)
    arrow(ax,(.247,.26),(.33,.26))
    arrow(ax,(.92,.605),(.475,.36))
    ax.text(.70,.49,'Context tokens supply keys and values',ha='center',fontsize=11,color=INK,
            bbox={'facecolor':'white','edgecolor':'none','pad':3})
    arrow(ax,(.607,.26),(.70,.26))
    fig.text(.035,.025,'“Global” means across the selected windows/series of one study. Pixels are encoded by the CNN first; this is not a patch-based Vision Transformer.',fontsize=12,color=INK)
    for ext in ('png','svg'):fig.savefig(OUT/f'transformer_detail.{ext}',dpi=160,facecolor='white')
    plt.close(fig)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    from local_global_model import SliceModel
    model=SliceModel('resnet50',False,.4,12,.1)
    counts={name:sum(p.numel() for p in module.parameters()) for name,module in model.named_children()}
    counts['queries']=model.queries.numel();counts['total']=sum(p.numel() for p in model.parameters())
    (OUT/'parameter_counts.json').write_text(json.dumps(counts,indent=2)+'\n')
    samples=sample();overview(samples);transformer()
    data={'samples':[{k:v for k,v in s.items() if k!='pixels'} for s in samples], 'counts':counts}
    p=OUT/'local-global-reader.html'
    text=(ROOT/'docs/figures/illustrations/local_global_reader.template.html').read_text()
    opening='<script type="application/json" id="klg-data">'
    before,rest=text.split(opening,1);_,after=rest.split('</script>',1)
    text=before+opening+json.dumps(data,ensure_ascii=False)+'</script>'+after
    p.write_text(text)
    print(json.dumps(counts,indent=2));print('Illustrations saved locally; no GPU, network, or gold labels used.')

if __name__=='__main__':main()
