"""Local photographic evidence; never a biometric or facial-hair classifier.

Compare regional contrast and edge texture at matched resolution. A smooth
original has measurable absolute texture even when a relative ratio is
undefined. Reference appearance is a soft tie-breaker only, not identity.
"""
from __future__ import annotations

import math
import cv2
import numpy as np

REGIONS = {
    "moustache": (.32,.51,.68,.60),
    "chin": (.30,.77,.70,.90),
    "jaw_l": (.18,.66,.33,.85),
    "jaw_r": (.67,.66,.82,.85),
    "lower_face": (.20,.72,.80,.92),
}
CHEEKS = ((.15,.43,.30,.59),(.70,.43,.85,.59))


def _bounds(bounds, shape):
    try:
        x,y,w,h = (float(bounds[k]) for k in ("x","y","width","height"))
    except (KeyError,TypeError,ValueError):
        return None
    if not all(math.isfinite(v) for v in (x,y,w,h)) or min(w,h)<40:
        return None
    if x<0 or y<0 or x+w>shape[1]+1 or y+h>shape[0]+1:
        return None
    return x,y,w,h


def mesh_bounds(points, shape):
    try:
        pts=np.asarray(points,np.float32)
    except (TypeError,ValueError):
        return None
    if pts.shape!=(478,2) or not np.isfinite(pts).all():
        return None
    # Face oval, not hair or neck. Reject strong tilt and side-on poses.
    left,right=pts[234],pts[454]
    top,chin=pts[10],pts[152]
    x=float(min(left[0],right[0])); w=float(abs(right[0]-left[0]))
    y=float(top[1]); h=float(chin[1]-top[1])
    if w<40 or h<40 or abs(float(left[1]-right[1]))>.16*h:
        return None
    if not x+.28*w<float(chin[0])<x+.72*w:
        return None
    b=dict(x=x,y=y,width=w,height=h)
    return b if _bounds(b,shape) else None


def _normalized_gray(image,bounds,edge):
    if image is None or image.ndim!=3 or image.shape[2]<3:
        return None
    b=_bounds(bounds,image.shape)
    if b is None: return None
    x,y,w,h=b
    crop=image[int(y):int(y+h),int(x):int(x+w),:3]
    if min(crop.shape[:2])<edge: return None
    gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray,(edge,edge),interpolation=cv2.INTER_AREA).astype(np.float32)


def _region(gray,box):
    h,w=gray.shape
    l,t,r,b=box
    return gray[int(t*h):int(b*h),int(l*w):int(r*w)]


def describe_regions(gray):
    if gray is None: return {}
    cheek=float(np.median(np.concatenate([_region(gray,b).ravel() for b in CHEEKS])))
    result={}
    for name,box in REGIONS.items():
        patch=_region(gray,box)
        if min(patch.shape)<5: continue
        smooth=cv2.GaussianBlur(patch,(0,0),1.0)
        high=patch-smooth
        # Local dark-detail fraction is a contrast proxy: a beard, shadow,
        # compression and other texture may all contribute.
        gx=cv2.Sobel(patch,cv2.CV_32F,1,0,ksize=3)/8.0
        gy=cv2.Sobel(patch,cv2.CV_32F,0,1,ksize=3)/8.0
        xx,yy,xy=float(np.mean(gx*gx)),float(np.mean(gy*gy)),float(np.mean(gx*gy))
        # Texture orientation coherence, not proof of facial hair. Only use
        # it where the native image actually contains sufficient gradients.
        coherence=math.hypot(xx-yy,2*xy)/(xx+yy) if xx+yy>=4 else None
        result[name]={
            "coherence":coherence,
            "energy":float(np.sqrt(np.mean(high**2))),
            "edge":float(np.mean(np.abs(high)>=3)),
            "dark":float(np.mean((smooth-patch)>=3)),
            "contrast":float(cheek-np.median(patch)),
        }
    return result


def reference_evidence(original, edited, bounds, landmarks=None, references=()):
    """Return absolute source/output measurements and optional reference fit.

    All pixels stay local. At least two valid reference crops are required;
    pose/lighting ambiguity means absent evidence, never a perfect score.
    """
    if original is None or edited is None or original.shape[:2]!=edited.shape[:2]:
        return {},None
    b=_bounds(bounds,original.shape)
    if b is None: return {},None
    edge=min(96,int(min(b[2:])))
    before=describe_regions(_normalized_gray(original,bounds,edge))
    after=describe_regions(_normalized_gray(edited,bounds,edge))
    evidence={name:(before[name]["energy"],after[name]["energy"],before[name]["edge"],after[name]["edge"])
              for name in before.keys() & after.keys()}
    # Use the same mesh-defined frame on the target and references for the
    # comparison, rather than comparing unrelated bounding-box conventions.
    target_box=mesh_bounds(landmarks,edited.shape)
    if target_box is None: return evidence,None
    edge=min(edge,int(target_box["width"]),int(target_box["height"]))
    rows=[]
    for image,points in references:
        if image is None: continue
        ref_box=mesh_bounds(points,image.shape)
        if ref_box is None: continue
        # Never invent high-resolution texture by upscaling a weak source.
        if min(ref_box["width"],ref_box["height"])<edge: continue
        row=describe_regions(_normalized_gray(image,ref_box,edge))
        if row: rows.append(row)
    if len(rows)<2: return evidence,None
    measured=describe_regions(_normalized_gray(edited,target_box,edge))
    distances=[]
    for name in ("moustache","chin","jaw_l","jaw_r"):
        if name not in measured or any(name not in row for row in rows): continue
        for metric,floor in (("energy",2.0),("edge",.10),("dark",.08),("contrast",12.0),("coherence",.20)):
            if measured[name][metric] is None or any(row[name][metric] is None for row in rows): continue
            values=np.array([row[name][metric] for row in rows],np.float32)
            low,high=float(np.min(values)),float(np.max(values))
            value=measured[name][metric]
            distances.append(max(0.,low-value,value-high)/max(floor,high-low))
    if len(distances)<8: return evidence,None
    return evidence, float(np.exp(-np.mean(np.clip(distances,0,4))))


def reference_evidence_file(original_path, edited_path, bounds, landmarks, real_face):
    original=cv2.imread(str(original_path),cv2.IMREAD_COLOR)
    edited=cv2.imread(str(edited_path),cv2.IMREAD_UNCHANGED)
    # Measure the RGB actually delivered by binary-alpha transport, not
    # undefined RGB hidden outside the authorized edit extent.
    if original is not None and edited is not None and edited.ndim==3 and edited.shape[2]==4 and original.shape[:2]==edited.shape[:2]:
        edited=np.where(edited[:,:,3:4]>0,edited[:,:,:3],original)
    refs=[]
    if real_face:
        for payload,ref in zip(real_face.get("images",()),real_face.get("references",())):
            image=cv2.imdecode(np.frombuffer(payload,np.uint8),cv2.IMREAD_COLOR)
            refs.append((image,ref.get("landmarks")))
    return reference_evidence(original,edited,bounds,landmarks,refs)


def format_evidence(evidence):
    return ",".join(f"{name}:{a:.2f}>{b:.2f}/{c:.3f}>{d:.3f}"
                    for name,(a,b,c,d) in sorted(evidence.items()))


def regional_tie_indices(identity, details, incumbent, sufficient=None, limit=3):
    """At most three candidates, including incumbent; preserve the peak band."""
    if not 0<=incumbent<len(identity) or len(identity)!=len(details) or limit<1:
        return []
    def valid(value):
        return isinstance(value,(float,int)) and not isinstance(value,bool) and math.isfinite(value)
    base=identity[incumbent]; quality=details[incumbent]
    if not valid(base) or not valid(quality):
        return []
    peak=max(value for value in identity if valid(value))
    threshold=max(base-.002,peak-.003)
    if valid(sufficient) and base>=sufficient:
        threshold=max(sufficient,base-.002)
    eligible=[i for i,(score,detail) in enumerate(zip(identity,details))
              if i!=incumbent and valid(score) and valid(detail)
              and score>=threshold and detail>=quality-.015]
    eligible.sort(key=lambda i:(identity[i],details[i]),reverse=True)
    return [incumbent]+eligible[:limit-1]


def choose_regional_evidence(identity, details, incumbent, scores, sufficient=None):
    allowed=regional_tie_indices(identity,details,incumbent,sufficient)
    if not allowed:
        return incumbent
    base=scores.get(incumbent)
    if not isinstance(base,(float,int)) or not math.isfinite(base) or not 0<=base<=1:
        return incumbent
    measured=[i for i in allowed if i in scores and isinstance(scores[i],(float,int))
              and math.isfinite(scores[i]) and 0<=scores[i]<=1 and scores[i]>=base+.075]
    return max(measured,key=lambda i:(scores[i],identity[i],details[i])) if measured else incumbent
