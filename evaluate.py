__author__ = "Alexey Antonenko, vedrusss@gmail.com"

from collections import defaultdict

def evaluate_classifier(predictions, labels, verbose=False):
    if verbose:
        for prediction, label in zip(predictions, labels):
            print(f"prediction: {prediction} || groundtruth label: {label}")
    pre, rec, f1s, integral_pre, integral_rec, integral_f1s = get_pre_rec_f1s(predictions, labels)
    print("Evaluation results:")
    for label in sorted(set(labels)):
        print(f"'{label}' :\tpre {pre.get(label)},\trec {rec.get(label)},\tf1s {f1s.get(label)}")
    print(f"Integral metrics: pre {integral_pre}, rec {integral_rec}, f1s {integral_f1s}")

def get_pre_rec_f1s(predictions, groundtruth, precision=3):
    tps, fps, fns = defaultdict(int), defaultdict(int), defaultdict(int)
    amount = defaultdict(int)
    for p, g in zip(predictions, groundtruth):
        amount[g] += 1
        if p == g:
            tps[g] += 1
        else:
            fps[p] += 1
            fns[g] += 1
    pre = {g : round(float(tps[g] / (tps[g] + fps[g])), precision) for g in amount.keys()}
    rec = {g : round(float(tps[g] / (tps[g] + fns[g])), precision) for g in amount.keys()}
    f1s = {g : round(2.*pre[g]*rec[g] / (pre[g] + rec[g]), precision) for g in amount.keys() if pre[g] or rec[g]}
    integral_pre = round(sum(pre.values()) / len(pre), precision)
    integral_rec = round(sum(rec.values()) / len(rec), precision)
    integral_f1s = round(sum(f1s.values()) / len(f1s), precision)
    return pre, rec, f1s, integral_pre, integral_rec, integral_f1s
