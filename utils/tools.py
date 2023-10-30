import numpy as np

def gen_label(labels, nb_label):
    num = len(labels)
    gt = np.zeros(shape=(num, nb_label))
    for i, label in enumerate(labels):
        gt[i, label] = 1
    return gt

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()

def convert_models_to_fp16(model):
    print(model)
    for p in model.parameters():
        p.data = p.data.half()
        p.grad.data = p.grad.data.half()
