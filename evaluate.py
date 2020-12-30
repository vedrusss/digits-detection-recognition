__author__ = "Alexey Antonenko, vedrusss@gmail.com"

def evaluate_classifier(predictions, groundtruth):
    tps, amount = {}, {}
    for p, g in zip(predictions, groundtruth):
        if not g in tps:
            tps[g] = 0
            amount[g] = 0
        if p == g: tps[g] += 1
        amount[g] += 1
    tps = {g : float(tps[g])/amount[g] for g in tps.keys()}
    integral_tps = sum(tps.values()) / len(tps)
    return tps, integral_tps