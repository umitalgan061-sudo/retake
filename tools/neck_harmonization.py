"""Conservative, local jaw-to-neck tone harmonization.

No neck geometry generation or reference-neck replacement. Only a narrow,
connected original-skin region below a validated chin mesh can be adjusted.
When visibility, pose or skin evidence is insufficient, return no candidate.
"""
from __future__ import annotations
import math
from pathlib import Path
import cv2
import numpy as np


def harmonize_neck(original, composite, bounds, landmarks):
    if (original is None or composite is None or original.ndim != 3
            or original.shape[2] != 3 or composite.shape != (*original.shape[:2],4)):
        return None, "unreadable", 0
    try:
        x,y,fw,fh = (float(bounds[k]) for k in ("x","y","width","height"))
        points = np.asarray(landmarks, dtype=np.float32)
    except (TypeError,ValueError,KeyError):
        return None,"no-mesh",0
    if (not all(math.isfinite(v) for v in (x,y,fw,fh)) or min(fw,fh)<48
            or points.shape != (478,2) or not np.isfinite(points).all()):
        return None,"no-mesh",0
    h,w = original.shape[:2]
    # MediaPipe chin and lower jaw. Avoid strongly tilted, ambiguous profiles.
    chin=points[152]
    left,right=points[172],points[397]
    cx,cy=map(float,chin)
    if not (x+.25*fw < cx < x+.75*fw and y+.65*fh < cy < y+1.12*fh
            and abs(float(left[1]-right[1])) < .22*fh
            and abs(float(left[0]-right[0])) > .32*fw):
        return None,"pose-uncertain",0
    # Restrict the corridor to upper visible neck; do not reach shoulders.
    x0=max(0,int(cx-.28*fw)); x1=min(w,int(cx+.28*fw)+1)
    y0=max(0,int(cy-.35*fh)); y1=min(h,int(cy+.35*fh)+1)
    if y1-int(cy)<max(12,int(.12*fh)) or x1-x0<16:
        return None,"neck-not-visible",0
    src=original[y0:y1,x0:x1]
    dst=composite[y0:y1,x0:x1]
    lab=cv2.cvtColor(src,cv2.COLOR_BGR2LAB).astype(np.float32)
    edited=cv2.cvtColor(dst[:,:,:3],cv2.COLOR_BGR2LAB).astype(np.float32)
    yy,xx=np.mgrid[y0:y1,x0:x1]
    # A skin anchor on both lower cheeks, excluding central beard/lips.
    anchors=[]
    for idx in (205,425):
        px,py=map(float,points[idx])
        if not (x<=px<=x+fw and y<=py<=cy): continue
        rad=max(2,int(.045*fw))
        ax0,ax1=max(0,int(px)-rad),min(w,int(px)+rad+1)
        ay0,ay1=max(0,int(py)-rad),min(h,int(py)+rad+1)
        a=original[ay0:ay1,ax0:ax1]
        if a.size: anchors.append(cv2.cvtColor(a,cv2.COLOR_BGR2LAB).reshape(-1,3))
    if len(anchors)!=2: return None,"skin-uncertain",0
    medians=[np.median(a,axis=0) for a in anchors]
    if np.linalg.norm(medians[0][1:]-medians[1][1:])>12:
        return None,"skin-uncertain",0
    skin=np.median(np.concatenate(anchors),axis=0)
    # Avoid neutral backgrounds/clothing: demand a measurable chroma anchor.
    if np.linalg.norm(skin[1:]-128)<5:
        return None,"skin-uncertain",0
    chroma=np.linalg.norm(lab[:,:,1:]-skin[1:],axis=2)
    skin_like=(chroma<10)&(np.abs(lab[:,:,0]-skin[0])<58)
    # Sample only actual edited lower-face skin to obtain measured correction.
    sample=skin_like & (yy>=cy-.30*fh)&(yy<=cy-.07*fh)&(dst[:,:,3]>180)
    if np.count_nonzero(sample)<max(18,int(.006*fw*fh)):
        return None,"skin-uncertain",0
    offset=np.median((edited-lab)[sample],axis=0)
    offset=np.clip(offset,[-10,-5,-5],[10,5,5])
    if np.linalg.norm(offset)<1.2: return None,"already-balanced",0
    rel=(yy-cy)/fh
    corridor=(rel>=0)&(rel<.34)&(np.abs(xx-cx)<fw*(.28-.12*rel/.34))
    # No editing through collars, hair or disconnected patches. Erode before
    # component selection to avoid bridges across thin boundaries.
    possible=(skin_like & corridor & (dst[:,:,3]==0)).astype(np.uint8)
    possible=cv2.erode(possible,np.ones((3,3),np.uint8))
    n,labels,stats,_=cv2.connectedComponentsWithStats(possible,8)
    seed=(rel>=.015)&(rel<=.13)&(np.abs(xx-cx)<.13*fw)
    viable=[k for k in range(1,n) if np.count_nonzero((labels==k)&seed)>=5]
    if not viable: return None,"skin-uncertain",0
    chosen=max(viable,key=lambda k:stats[k,cv2.CC_STAT_AREA])
    mask=(labels==chosen).astype(np.uint8)
    if np.count_nonzero(mask)<max(24,int(.015*fw*fh)):
        return None,"neck-not-visible",0
    inward=cv2.distanceTransform(mask,cv2.DIST_L2,5)
    feather=np.clip(inward/max(2.,fw*.045),0,1)
    falloff=np.sin(np.pi*np.clip(rel/.34,0,1))**.7
    weight=(feather*falloff*.65).astype(np.float32)
    adjusted=cv2.cvtColor(np.clip(np.rint(lab+offset*weight[:,:,None]),0,255).astype(np.uint8),cv2.COLOR_LAB2BGR)
    # Preserve the exact original RGB anywhere outside the validated mask,
    # including color conversion roundoff; existing face pixels never change.
    changed=(weight>.04)&(np.max(cv2.absdiff(adjusted,src),axis=2)>=2)
    if np.count_nonzero(changed)<max(12,int(.005*fw*fh)):
        return None,"already-balanced",0
    output=composite.copy()
    roi=output[y0:y1,x0:x1]
    roi[changed,:3]=adjusted[changed]
    roi[changed,3]=255
    return output,"applied",int(np.count_nonzero(changed))


def neck_candidate_file(original_path, composite_path, output_path, bounds, landmarks):
    original=cv2.imread(str(original_path),cv2.IMREAD_COLOR)
    composite=cv2.imread(str(composite_path),cv2.IMREAD_UNCHANGED)
    result,status,count=harmonize_neck(original,composite,bounds,landmarks)
    if result is not None and not cv2.imwrite(str(output_path),result):
        raise OSError("Neck candidate write failed")
    return status,count
