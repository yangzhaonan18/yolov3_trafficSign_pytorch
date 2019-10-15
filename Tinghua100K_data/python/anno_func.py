import json
import pylab as pl
import random
import numpy as np
import cv2
import copy

# type45="i2,i4,i5,il100,il60,il80,io,ip,p10,p11,p12,p19,p23,p26,p27,p3,p5,p6,pg,ph4,ph4.5,ph5,pl100,pl120,pl20,pl30,pl40,pl5,pl50,pl60,pl70,pl80,pm20,pm30,pm55,pn,pne,po,pr40,w13,w32,w55,w57,w59,wo"

type_all = "w55,i13,pl1,w31,p27,il80,w32,i1,il100,pl90,w3,w34,w66,i16,il50,pr70,w56,il70,pdc,pl10,w39,il90,pl70,pl30,pl55,i3,i11,pl35,pr30,w43,pr0,w41,pb,i9,il110,p20,p11,pr20,w13,w42,pl2,pn,w21,w57,zo,pl60,p10,p23,w68,w15,i2,pg,pl5,pl120,pa,pl110,ps,il60,pr80,w24,pl8,p14,i12,p6,p9,pl100,pr40,w16,w45,i14,p28,pr100,pl4,w30,pl50,p31,i5,i10,i18,p19,i17,i19,p29,w69,p21,ip,p12,pr60,pl25,w46,pnl,pl7,pl20,p30,w10,i6,pr50,w20,w47,p13,pne,w63,p5,w22,pl130,p32,i4,pl3,pl40,pl80,w40,pl15,ipp,p1,i15"
type_no = "i15,ipp,w40,pl3,pl130,p32,i6,w10,p30,pl7,pnl,w69,,p29,i19,i17,i18,p31,pl4,pl8,w24,pa,w68,zo,pl2,i9,pr0,w43,pl55,w39,pdc,w56,i16,w66,w31,pl1"
type_few = "i13,w32,w34,pl10,il90,pl35,w41,w21,p10,w15,i2,pl5,i12,p6,p9,w16,w45,p28,pr100,i10,p12,w46,w20,w47,p13,w63,i4,"       
# type45 = type45.split(',')
type_all = type_all.split(',')
type_no = type_no.split(',')
type_few = type_few.split(',')

type45 = [i for i in type_all if (i not in type_no and i not in type_few)]
print("type = ", len(type45))

def load_img(annos, datadir, imgid):
    img = annos["imgs"][imgid]
    imgpath = datadir+'/'+img['path']
    # imgdata = pl.imread(imgpath)

    imgdata = cv2.imread(imgpath)
    #imgdata = (imgdata.astype(np.float32)-imgdata.min()) / (imgdata.max() - imgdata.min())
    if imgdata.max() > 2:
        imgdata = imgdata/255.
    return imgdata

def load_mask(annos, datadir, imgid, imgdata):
    img = annos["imgs"][imgid]
    mask = np.zeros(imgdata.shape[:-1])
    mask_poly = np.zeros(imgdata.shape[:-1])
    mask_ellipse = np.zeros(imgdata.shape[:-1])
    for obj in img['objects']:
        box = obj['bbox']
        cv2.rectangle(mask, (int(box['xmin']), int(box['ymin'])), (int(box['xmax']), int(box['ymax'])), 1, -1)
        # if obj.has_key('polygon') and len(obj['polygon'])>0:
        if 'polygon' in obj and len(obj['polygon'])>0:
            pts = np.array(obj['polygon'])
            cv2.fillPoly(mask_poly, [pts.astype(np.int32)], 1)
            # print pts
        else:
            cv2.rectangle(mask_poly, (int(box['xmin']), int(box['ymin'])), (int(box['xmax']), int(box['ymax'])), 1, -1)
        # if obj.has_key('ellipse'):
        if 'ellipse' in obj:
            rbox = obj['ellipse']
            rbox = ((rbox[0][0], rbox[0][1]), (rbox[1][0], rbox[1][1]), rbox[2])
            print( rbox)
            cv2.ellipse(mask_ellipse, rbox, 1, -1)
        else:
            cv2.rectangle(mask_ellipse, (int(box['xmin']), int(box['ymin'])), (int(box['xmax']), int(box['ymax'])), 1, -1)
    mask = np.multiply(np.multiply(mask,mask_poly),mask_ellipse)
    return mask
    
def draw_all(annos, datadir, imgid, imgdata, color=(0,1,0), have_mask=True, have_label=True):
    img = annos["imgs"][imgid]
    if have_mask:
        mask = load_mask(annos, datadir, imgid, imgdata)
        imgdata = imgdata.copy()
        imgdata[:,:,0] = np.clip(imgdata[:,:,0] + mask*0.7, 0, 1)
    for obj in img['objects']:
        box = obj['bbox']
        cv2.rectangle(imgdata, (int(box['xmin']), int(box['ymin'])), (int(box['xmax']), int(box['ymax'])), color, 3)
        ss = obj['category']
        # if obj.has_key('correct_catelog'):
        if 'correct_catelog' in obj:
            ss = ss+'->'+obj['correct_catelog']
        if have_label:
            cv2.putText(imgdata, ss, (int(box['xmin']),int(box['ymin']-10)), 0, 1, color, 2)
    return imgdata

def rect_cross(rect1, rect2):
    rect = [max(rect1[0], rect2[0]),
            max(rect1[1], rect2[1]),
            min(rect1[2], rect2[2]),
            min(rect1[3], rect2[3])]
    rect[2] = max(rect[2], rect[0])
    rect[3] = max(rect[3], rect[1])
    return rect

def rect_area(rect):
    return float(max(0.0, (rect[2]-rect[0])*(rect[3]-rect[1])))

def calc_cover(rect1, rect2):
    crect = rect_cross(rect1, rect2)
    return rect_area(crect) / rect_area(rect2)

def calc_iou(rect1, rect2):
    crect = rect_cross(rect1, rect2)
    ac = rect_area(crect)
    a1 = rect_area(rect1)
    a2 = rect_area(rect2)
    return ac / (a1+a2-ac)

def get_refine_rects(annos, raw_rects, minscore=20):
    cover_th = 0.5
    refine_rects = {}

    for imgid in raw_rects.keys():
        v = raw_rects[imgid]
        tv = copy.deepcopy(sorted(v, key=lambda x:-x[2]))
        nv = []
        for obj in tv:
            rect = obj[1]
            rect[2]+=rect[0]
            rect[3]+=rect[1]
            if rect_area(rect) == 0: continue
            if obj[2] < minscore: continue
            cover_area = 0
            for obj2 in nv:
                cover_area += calc_cover(obj2[1], rect)
            if cover_area < cover_th:
                nv.append(obj)
        refine_rects[imgid] = nv
    results = {}
    for imgid, v in refine_rects.items():
        objs = []
        for obj in v:
            mobj = {"bbox":dict(zip(["xmin","ymin","xmax","ymax"], obj[1])), 
                    "category":annos['types'][int(obj[0]-1)], "score":obj[2]}
            objs.append(mobj)
        results[imgid] = {"objects":objs}
    results_annos = {"imgs":results}
    return results_annos

def box_long_size(box):
    return max(box['xmax']-box['xmin'], box['ymax']-box['ymin'])

def eval_annos(annos_gd, annos_rt, iou=0.75, imgids=None, check_type=True, types=None, minscore=40, minboxsize=0, maxboxsize=400, match_same=True):
    ac_n, ac_c = 0,0
    rc_n, rc_c = 0,0
    if imgids==None:
        imgids = annos_rt['imgs'].keys()
    if types!=None:
        types = { t:0 for t in types }
    miss = {"imgs":{}}
    wrong = {"imgs":{}}
    right = {"imgs":{}}
    
    for imgid in imgids:
        v = annos_rt['imgs'][imgid]
        vg = annos_gd['imgs'][imgid]
        convert = lambda objs: [ [ obj['bbox'][key] for key in ['xmin','ymin','xmax','ymax']] for obj in objs]
        objs_g = vg["objects"]
        objs_r = v["objects"]
        bg = convert(objs_g)
        br = convert(objs_r)
        
        match_g = [-1]*len(bg)
        match_r = [-1]*len(br)
        if types!=None:
            for i in range(len(match_g)):
                # if not types.has_key(objs_g[i]['category']):
                if not objs_g[i]['category'] in types:
                    match_g[i] = -2
            for i in range(len(match_r)):
                # if not types.has_key(objs_r[i]['category']):
                if not objs_r[i]['category'] in types:
                    match_r[i] = -2
        for i in range(len(match_r)):
            # if objs_r[i].has_key('score') and objs_r[i]['score']<minscore:
            if 'score' in objs_r[i] and objs_r[i]['score']<minscore:
                match_r[i] = -2
        matches = []
        for i,boxg in enumerate(bg):
            for j,boxr in enumerate(br):
                if match_g[i] == -2 or match_r[j] == -2:
                    continue
                if match_same and objs_g[i]['category'] != objs_r[j]['category']: continue
                tiou = calc_iou(boxg, boxr)
                if tiou>iou:
                    matches.append((tiou, i, j))
        matches = sorted(matches, key=lambda x:-x[0])
        for tiou, i, j in matches:
            if match_g[i] == -1 and match_r[j] == -1:
                match_g[i] = j
                match_r[j] = i
                
        for i in range(len(match_g)):
            boxsize = box_long_size(objs_g[i]['bbox'])
            erase = False
            if not (boxsize>=minboxsize and boxsize<maxboxsize):
                erase = True
            #if types!=None and not types.has_key(objs_g[i]['category']):
            #    erase = True
            if erase:
                if match_g[i] >= 0:
                    match_r[match_g[i]] = -2
                match_g[i] = -2
        
        for i in range(len(match_r)):
            boxsize = box_long_size(objs_r[i]['bbox'])
            if match_r[i] != -1: continue
            if not (boxsize>=minboxsize and boxsize<maxboxsize):
                match_r[i] = -2
                    
        miss["imgs"][imgid] = {"objects":[]}
        wrong["imgs"][imgid] = {"objects":[]}
        right["imgs"][imgid] = {"objects":[]}
        miss_objs = miss["imgs"][imgid]["objects"]
        wrong_objs = wrong["imgs"][imgid]["objects"]
        right_objs = right["imgs"][imgid]["objects"]
        
        tt = 0
        for i in range(len(match_g)):
            if match_g[i] == -1:
                miss_objs.append(objs_g[i])
        for i in range(len(match_r)):
            if match_r[i] == -1:
                obj = copy.deepcopy(objs_r[i])
                obj['correct_catelog'] = 'none'
                wrong_objs.append(obj)
            elif match_r[i] != -2:
                j = match_r[i]
                obj = copy.deepcopy(objs_r[i])
                if not check_type or objs_g[j]['category'] == objs_r[i]['category']:
                    right_objs.append(objs_r[i])
                    tt+=1
                else:
                    obj['correct_catelog'] = objs_g[j]['category']
                    wrong_objs.append(obj)
                    
        
        rc_n += len(objs_g) - match_g.count(-2)
        ac_n += len(objs_r) - match_r.count(-2)
        
        ac_c += tt
        rc_c += tt
    if types==None:
        styps = "all"
    elif len(types)==1:
        styps = list(types.keys())[0]
    elif not check_type or len(types)==0:
        styps = "none"
    else:
        styps = "[%s, ...total %s...]"%(list(types.keys())[0], len(types))
    report = "iou:%s, size:[%s,%s), types:%s, accuracy:%s, recall:%s"% (
        iou, minboxsize, maxboxsize, styps, 1 if ac_n==0 else ac_c*1.0/ac_n, 1 if rc_n==0 else rc_c*1.0/rc_n)
    summury = {
        "iou":iou,
        "accuracy":1 if ac_n==0 else ac_c*1.0/ac_n,
        "recall":1 if rc_n==0 else rc_c*1.0/rc_n,
        "miss":miss,
        "wrong":wrong,
        "right":right,
        "report":report
    }
    return summury
