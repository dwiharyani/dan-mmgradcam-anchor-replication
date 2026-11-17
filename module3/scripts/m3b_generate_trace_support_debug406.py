#!/usr/bin/env python3
"""Compute spatial waveform-corridor support from the SAME Module2 re-rendered 2D CNN pipeline.

For each ECG, spatial H2D(y,u) is the unperturbed 2D CAM (max-normalized,
then bilinearly resized to 300 x 600); M(y,u) is the original Module2 ECG
waveform mask dilated by 6 pixels. Define per-column support as
F_trace(u) = sum_y H2D(y,u) M(y,u) / (sum_y H2D(y,u) + eps).
This factor is in [0,1], is NOT re-normalized by its peak, and is computed
from the same re-rendered baseline 2D model/image as Module2. It is a spatial
SUPPORT HEURISTIC, not a statistical reliability guarantee.

This performs new CNN 2D inference and Grad-CAM, but NO perturbation reruns.
Must run in a compute allocation and pass exact Module2 2D baseline checks.
"""
import argparse,os,sys,json,importlib.util
from pathlib import Path
import numpy as np,pandas as pd

PROVENANCE='module2_rerendered_baseline_2d_spatial'

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--m2-dir',required=True)
    ap.add_argument('--m3a-dir',required=True)
    ap.add_argument('--runtime-root',required=True)
    ap.add_argument('--module2-common',required=True)
    ap.add_argument('--only-ids',default=None,help='Comma-separated subset, e.g. 209')
    ap.add_argument('--start',type=int,default=0,help='start offset within M3A eligible IDs')
    ap.add_argument('--limit',type=int,default=0,help='Number of records after start, 0=all')
    ap.add_argument('--atol',type=float,default=1e-5)
    ap.add_argument('--rtol',type=float,default=1e-5)
    ap.add_argument('--cam-atol',type=float,default=0.005,help='Engineering tolerance to reproduce saved M2 projected base2 CAM')
    ap.add_argument('--cam-rtol',type=float,default=0.001)
    ap.add_argument('--out',required=True,help='NPZ file path, use unique name for chunks')
    a=ap.parse_args()
    if not Path(a.module2_common).expanduser().is_file():raise FileNotFoundError(a.module2_common)
    folder=Path(a.runtime_root).expanduser().resolve()
    for path in ['data/ECGData','models/MMGradCAM_2d_trained_model.keras','results/module2/standard_scaler.npz']:
        if not (folder/path).exists():raise FileNotFoundError(f'Missing runtime input: {folder/path}')
    m2=Path(a.m2_dir).expanduser();m3a=Path(a.m3a_dir).expanduser()
    src=np.load(m2/'noise_stability_maps.npz',allow_pickle=False)
    try:
        mids=src['cohort_index'].astype(int)
        midsmap={int(k):i for i,k in enumerate(mids)}
        classes=pd.read_csv(m2/'noise_stability_records.csv')
        if not np.array_equal(classes.cohort_index.to_numpy(dtype=int),mids):raise ValueError('Module2 record ordering mismatch')
        m3=np.load(m3a/'module3a_maps.npz',allow_pickle=False)
        try:
            ids=m3['cohort_index'].astype(int)
            if a.only_ids:
                want=set(int(s.strip()) for s in a.only_ids.split(',') if s.strip())
                unknown=want-set(ids)
                if unknown:raise ValueError(f'Requested IDs not M3A eligible: {sorted(unknown)}')
                ids=np.asarray([v for v in ids if v in want],dtype=int)
            else:
                ids=ids[a.start:] if a.limit==0 else ids[a.start:a.start+a.limit]
            if len(ids)==0:raise ValueError('Empty ID selection')
            valid_lookup={int(k):m3['valid_600'][i].astype(bool) for i,k in enumerate(m3['cohort_index'])}
        finally:m3.close()
        # IMPORT FROM EXACT ORIGINAL MODULE 2 CODE: it initializes TensorFlow,
        # checkpoints and renderer using the configured isolated runtime root.
        old=os.getcwd()
        os.chdir(folder)
        fp=Path(a.module2_common).expanduser().resolve()
        sys.path.insert(0,str(fp.parent))
        try:
            spec=importlib.util.spec_from_file_location('m3_original_module2_common',str(fp))
            cm=importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cm)
        except Exception:
            os.chdir(old)
            raise
        try:
            for required in ['Renderer','load_ecg','to_2d','run_model','gradcam','trace_mask_at_2d_grid','m2','L2','CLASSES','MASK_DILATE_Y']:
                if not hasattr(cm,required):raise ValueError(f'Module2 common missing function/variable: {required}')
            import tensorflow as tf
            renderer=cm.Renderer()
            support=[];errors=[];cam_errors=[];selected=[];diagnostics=[];selected_valid=[]
            for index,cid in enumerate(ids):
                row=classes.iloc[midsmap[cid]];raw=cm.load_ecg(str(row.ecg_csv_filename))
                rgb=renderer.base(raw)
                img=(cm.to_2d(rgb[None])/255.0).astype(np.float32)
                if int(cid) == 406:
                    import hashlib
                    print("DBG406 RGB_SHA:",
                          hashlib.sha256(rgb.tobytes()).hexdigest(), flush=True)
                    print("DBG406 INPUT_SHA:",
                          hashlib.sha256(img.tobytes()).hexdigest(), flush=True)

                prob=np.asarray(cm.run_model('2d',cm.m2,img)[0],dtype=np.float64)
                ref=np.asarray(src['out2_base'][midsmap[cid]],dtype=np.float64)
                err=float(np.max(np.abs(prob-ref)))
                if int(cid) == 406:
                    print("DBG406 PROB:", prob.tolist(), flush=True)
                    print("DBG406 REF:", ref.tolist(), flush=True)
                    print("DBG406 ERROR:", err, flush=True)

                pred=str(cm.CLASSES[int(np.argmax(prob))]);target=str(row.class_name)
                if not np.allclose(prob,ref,atol=a.atol,rtol=a.rtol):
                    raise RuntimeError(f'ECG {cid} 2D baseline probabilities mismatch: max error={err:.6g}. STOP: do not mix pipelines')
                if pred!=target:raise RuntimeError(f'ECG {cid}: 2D prediction {pred} != true target {target}; not eligible')
                native=cm.gradcam('2d',cm.m2,cm.L2,img,int(np.argmax(prob)))[0]
                v=valid_lookup[cid]
                reproduced_cam,_=cm.project_2d(native,rgb,v)
                saved_cam=np.asarray(src['base2'][midsmap[cid]],dtype=np.float64)
                camerr=float(np.max(np.abs(np.asarray(reproduced_cam)[v]-saved_cam[v])))
                if not np.allclose(np.asarray(reproduced_cam)[v],saved_cam[v],atol=a.cam_atol,rtol=a.cam_rtol):
                    raise RuntimeError(f'ECG {cid} 2D Grad-CAM baseline projection mismatch: max error={camerr:.6g}; stop before Module3B')
                c=np.asarray(native,dtype=np.float64)
                c=c/max(1e-12,float(c.max()))
                heat=tf.image.resize(c[...,None].astype(np.float32),[300,600],method='bilinear').numpy()[...,0]
                heat=np.clip(heat,0,1)
                mask=np.asarray(cm.trace_mask_at_2d_grid(rgb,dilate_y=cm.MASK_DILATE_Y),dtype=bool)
                if heat.shape!=(300,600) or mask.shape!=(300,600):raise ValueError(f'Unexpected spatial shapes for {cid}')
                denominator=np.sum(heat,axis=0,dtype=np.float64)
                numerator=np.sum(heat*mask,axis=0,dtype=np.float64)
                fraction=np.divide(numerator,denominator,out=np.zeros(600,dtype=np.float64),where=denominator>1e-12)
                result=np.full(600,np.nan,dtype=np.float32)
                result[v]=np.clip(fraction[v],0,1).astype(np.float32)
                selected.append(cid);selected_valid.append(v)
                support.append(result);errors.append(err);cam_errors.append(camerr)
                diagnostics.append({'cohort_index':int(cid),'class_name':target,'baseline_2d_max_error':err,'baseline_2d_cam_max_error':camerr,'support_mean_valid':float(result[v].mean()),'support_median_valid':float(np.median(result[v])),'zero_spatial_mass_columns':int(np.sum(denominator[v]<=1e-12)),'n_mask_pixels_mean':float(mask[:,v].sum(axis=0).mean())})
                if (index+1)%10==0 or index+1==len(ids):print(f'Computed F_trace for {index+1}/{len(ids)} ECGs; last ID={cid}',flush=True)
            out=Path(a.out).expanduser();out.parent.mkdir(parents=True,exist_ok=True)
            if out.exists():raise FileExistsError(f'Refusing to overwrite original trace-support output: {out}')
            np.savez_compressed(out,cohort_index=np.asarray(selected,dtype=np.int64),F_trace=np.stack(support),valid_600=np.stack(selected_valid),origin=np.asarray(PROVENANCE),baseline_2d_max_error=np.asarray(errors),baseline_2d_cam_max_error=np.asarray(cam_errors))
            pd.DataFrame(diagnostics).to_csv(out.with_suffix('.csv'),index=False)
            out.with_suffix('.json').write_text(json.dumps({'provenance':PROVENANCE,'formula':'sum_y(spatialCAM(y,u)*waveform_mask(y,u)) / sum_y(spatialCAM(y,u))','model':'Module2 rerendered original 2D CNN, exact saved 2D baseline probability check','n_ecgs':len(selected),'module2_source_dir':str(m2),'max_2d_prediction_error':max(errors),'max_2d_cam_projection_error':max(cam_errors),'uses_perturbed_cam':False,'limitations':['Spatial support is based on unperturbed Module2 spatial CAM, not perturbed spatial CAM','Spatial support is a heuristic and not a direct measurement of explanation faithfulness','Avoid mixing with original Module1 rendered-image CAM for primary comparisons']},indent=2))
            print('OUTPUT:',out)
        finally:
            try:
                import matplotlib.pyplot as plt
                plt.close(renderer.fig)
            except Exception:pass
            os.chdir(old)
    finally:src.close()

if __name__=='__main__':main()
