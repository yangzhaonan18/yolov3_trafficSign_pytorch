import json
import pylab as pl
import random
import numpy as np
import cv2
import anno_func

datadir = "../data/"

filedir = datadir + "annotations.json"

# ids = open(datadir + "/test/ids.txt").read().splitlines()

annos = json.loads(open(filedir).read())

# result_anno_file = "./../results/ours_result_annos.json"
# result_anno_file = "./../results/yangzhaonan.json"
result_anno_file = "/headless/Desktop/yzn_file/code/my_github/yolov3_trafficSign_pytorch/results/Tinghua100K_result.json"

results_annos = json.loads(open(result_anno_file).read())
sm = anno_func.eval_annos(annos, results_annos, iou=0.5, types=anno_func.type45, check_type=True)
# print( sm['report'])



# import sys
# def get_acc_res(results_annos, **argv):
#     scs = [ obj['score'] for k, img in results_annos['imgs'].items() for obj in img['objects']]
#     scs = sorted(scs)
#     accs = [0]
#     recs = [1]
#     for i, score in enumerate(np.linspace(0, scs[-1], 100)):
#         sm = anno_func.eval_annos(annos, results_annos, iou=0.5, check_type=True, types=anno_func.type45, minscore=score, **argv)
#         print( "\r%s %s %s" % (i, score, sm['report']))
#         sys.stdout.flush()
#         accs.append(sm['accuracy'])
#         if len(accs)>=2 and accs[-1]<accs[-2]:
#             accs[-1] = accs[-2]
#         recs.append(sm['recall'])
#     accs.append(1)
#     recs.append(0)
#     return accs, recs
# sizes = [0,32,96,400]
# ac_rc = []

# for i in range(4):
#     if i==3:
#         l=sizes[0]
#         r=sizes[-1]
#     else:
#         l=sizes[i]
#         r=sizes[i+1]
#     acc1, recs1 = get_acc_res(results_annos, minboxsize=l, maxboxsize=r)
#     #acc2, recs2 = get_acc_res(results_annos2, minboxsize=l, maxboxsize=r)
#     #ac_rc.append([acc1, recs1, acc2, recs2])
#     ac_rc.append([acc1, recs1])
    
#     pl.figure()
#     pl.plot(acc1, recs1, label='ours')
#     #pl.plot(acc2, recs2, label='fast-rcnn')
#     pl.xlabel("accuracy")
#     pl.ylabel("recall")
#     pl.title("size: (%s,%s]"%(l,r))

#     pl.legend(bbox_to_anchor=(0, 0), loc="lower left")
#     #pl.savefig("/home/randon/data/newdata/results/ac-rc%s.pdf"%i)
# #_ = pl.hist(scs, bins=100)




test_annos = results_annos
minscore=40

sm = anno_func.eval_annos(annos, test_annos, iou=0.5, check_type=True, types=anno_func.type45,
                         minboxsize=0,maxboxsize=400,minscore=minscore)
print(sm['report'])
sm = anno_func.eval_annos(annos, test_annos, iou=0.5, check_type=True, types=anno_func.type45,
                         minboxsize=0,maxboxsize=32,minscore=minscore)
print(sm['report'])
sm = anno_func.eval_annos(annos, test_annos, iou=0.5, check_type=True, types=anno_func.type45,
                         minboxsize=32,maxboxsize=96,minscore=minscore)
print(sm['report'])
sm = anno_func.eval_annos(annos, test_annos, iou=0.5, check_type=True, types=anno_func.type45,
                         minboxsize=96,maxboxsize=400,minscore=minscore)
print(sm['report'])

for tp in anno_func.type45:
    sm = anno_func.eval_annos(annos, test_annos, iou=0.5, check_type=True, types=[tp],minscore=minscore)
    print(sm['report'])
