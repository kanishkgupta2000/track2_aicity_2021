# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import logging
import time
import torch
import torch.nn as nn
import tqdm
import pickle
from lib.utils.reid_eval import evaluator
from os.path import exists


def inference(
        cfg,
        model,
        val_loader,
        num_query,
        dataset,
        run_batch
):

    device = cfg.MODEL.DEVICE
    logger = logging.getLogger("reid_baseline.inference")

    inference_file='inferences_'+cfg.DATASETS.TEST[0]+'.txt'
    if exists(inference_file):
        print("inferences file already exists, loading metric")
        infile=open(inference_file,'rb')
        metric=pickle.load(infile)
        metric.do_rerank=cfg.TEST.DO_RERANK
        infile.close()
    else:
        model.to(device)
        logger.info("Enter inferencing")
        print(f'num of query: {num_query}')
        metric = evaluator(num_query, dataset, cfg, max_rank=50)
        print("model eval")
        model.eval()
        start = time.time()  
        with torch.no_grad():
            for _,batch in enumerate(tqdm.tqdm(val_loader)):
                data, pid, camid, img_path = batch
                data = data.cuda()
                feats = model(data)
                if cfg.TEST.FLIP_TEST:
                    data_flip = data.flip(dims=[3])  # NCHW
                    feats_flip = model(data_flip)
                    feats = (feats + feats_flip) / 2
                output = [feats, pid, camid, img_path]
                metric.update(output)
        end = time.time()
        with open(inference_file, 'wb') as fh:
            pickle.dump(metric, fh)
        logger.info("inference takes {:.3f}s".format((end - start)))
    torch.cuda.empty_cache()
    cmc, mAP, indices_np = metric.compute(run_batch)
    #either generate 10 new evals from the eval object,
    # OR run new inferences to make 10 new objects
    #result is getting 10 obs,
    #run compute
    #get 10-11 results
    #agregate
    
    logger.info('Validation Results')
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return indices_np

def select_topk(indices, query, gallery, topk=10):
    results = []
    for i in range(indices.shape[0]):
        ids = indices[i][:topk]
        results.append([query[i][0]] + [gallery[id][0] for id in ids])
    return results


def extract_features(cfg, model, loader):
    device = cfg.MODEL.DEVICE
    model.to(device)
    model.eval()
    feats = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            data, pid, camid, img_path = batch
            data = data.cuda()
            feat = model(data)
            feats.append(feat)
    feats = torch.cat(feats, dim=0)
    feats = torch.nn.functional.normalize(feats, dim=1, p=2)
    return feats