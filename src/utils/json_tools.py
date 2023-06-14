import json

def dump_mono_model(cm1,cd1,path):
    d = {}
    d["cm1"] = cm1.tolist()
    d["cd1"] = cd1.tolist()
    with open(path,"w") as f:
        json.dump(d,f)
def dump_stereo_model(cm1,cd1,cm2,cd2,R,T,path):
    d = {}
    d["image_size"] = [4032,3040]
    d["is_fisheye"] = False
    d["cm1"] = cm1.tolist()
    d["cd1"] = cd1.tolist()
    d["cm2"] = cm2.tolist()
    d["cd2"] = cd2.tolist()
    d["R"] = R.tolist()
    d["T"] = T.T[0].tolist()
    with open(path,"w") as f:
        json.dump(d,f)