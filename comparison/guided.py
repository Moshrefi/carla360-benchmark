"""Guarded notebook workflow; one selected model, reusable completed outputs."""
import json,os,shutil,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
from comparison.prepare import prepare_job
from comparison.view import load_results,MODELS

ROOT=Path(__file__).resolve().parents[1]

def validate_settings(sequence,job,frame_ids,model):
    sequence=Path(sequence).expanduser().resolve();job=Path(job).expanduser().resolve()
    if model not in MODELS:raise ValueError(f'Choose one model from {MODELS}')
    if not (sequence/'poses.csv').is_file():raise ValueError(f'Sequence not found: {sequence}')
    if job==sequence or job.is_relative_to(sequence):raise ValueError('JOB must be outside the dataset')
    if (job/'manifest.json').exists():
        mf=json.loads((job/'manifest.json').read_text())
        if mf['source_sequence']!=str(sequence) or [r['frame_id'] for r in mf['frames']]!=list(frame_ids):
            raise ValueError('This JOB belongs to another selection. Change JOB_NAME; existing outputs preserved.')
        load_results(job)  # Check hashes, shapes and frame identities of completed outputs.
    elif job.exists() and any(job.iterdir()):
        raise ValueError('JOB is not empty and has no manifest. Choose a new JOB_NAME.')
    return sequence,job

def prepare(sequence,job,frame_ids):
    if not (Path(job)/'manifest.json').exists():prepare_job(sequence,job,frame_ids)
    return json.loads((Path(job)/'manifest.json').read_text())

def status(job):
    mf,results=load_results(job)
    return [{'model':m,'status':'completed' if m in results else 'incomplete' if (Path(job)/'predictions'/m).exists() else 'not run',
             'frames':len(mf['frames'])} for m in MODELS]

def command_prefix(environment):
    if environment is None or environment=='':return [sys.executable]
    env=str(environment)
    if '/' in env:
        p=Path(env).expanduser()
        binary=p/'bin/python' if p.is_dir() else p
        if not binary.is_file():raise ValueError(f'Python not found: {binary}')
        return [str(binary)]
    conda=shutil.which('conda') or os.environ.get('CONDA_EXE')
    if not conda:raise ValueError('Conda not found. Set ENVIRONMENT to the full environment path or Python executable.')
    return [conda,'run','--no-capture-output','-n',env,'python']

def preflight(job,model,environment,checkpoint):
    if model in load_results(job)[1]:
        print(f'{model}: completed result will be reused. No environment change needed.');return None
    prefix=command_prefix(environment)
    if model=='panovggt' and (not checkpoint or not Path(checkpoint).expanduser().is_file()):
        raise ValueError('Set CHECKPOINT to your existing PanoVGGT weight file.')
    if checkpoint and model in ('panovggt','omnivggt') and not Path(checkpoint).expanduser().is_file():
        raise ValueError(f'Checkpoint not found: {checkpoint}')
    if model=='panovggt':path=ROOT/'third_party/PanoVGGT'
    else:
        spec=json.loads((ROOT/'comparison/sources.json').read_text())[model];path=ROOT/'third_party'/spec['folder']
    if not path.is_dir():raise ValueError('Upstream sources missing. Run python -m comparison.setup_sources in the repository terminal.')
    check="import sys, torch; print('Python:',sys.executable); print('PyTorch:',torch.__version__); assert torch.cuda.is_available(), 'CUDA unavailable'; print('GPU:',torch.cuda.get_device_name(0))"
    modules={'vggt360':['torchvision','cv2','scipy','huggingface_hub'],
             'panovggt':['torchvision','cv2','omegaconf'],
             'omnivggt':['torchvision','safetensors','viser','onnxruntime'],
             'mapanything':['mapanything','scipy']}[model]
    check+='; '+ '; '.join('import '+m for m in modules)
    result=subprocess.run(prefix+['-c',check],cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    print(result.stdout)
    if result.returncode:raise RuntimeError('Environment check failed. Fix the reported dependency/GPU issue before running.')
    return prefix

def run(job,model,environment=None,checkpoint=None,retry_incomplete=True):
    job=Path(job);prefix=preflight(job,model,environment,checkpoint)
    if prefix is None:return
    target=job/'predictions'/model
    if target.exists():
        if not retry_incomplete:raise RuntimeError('Incomplete attempt exists. Set RETRY_INCOMPLETE=True to archive and retry.')
        backup=job/'failed_attempts'/f'{model}_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")}'
        backup.parent.mkdir(parents=True,exist_ok=True);target.rename(backup)
        print('Previous incomplete attempt preserved:',backup)
    command=prefix+['-m','comparison.run','--job',str(job),'--model',model]
    if checkpoint and model in ('panovggt','omnivggt'):command+=['--checkpoint',str(Path(checkpoint).expanduser())]
    logs=job/'logs';logs.mkdir(exist_ok=True)
    log=logs/f'{model}_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")}.txt'
    print('Running only:',model,flush=True)
    with log.open('w') as f:
        process=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        try:
            for line in process.stdout:print(line,end='',flush=True);f.write(line);f.flush()
            rc=process.wait()
        except BaseException:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            raise
    if rc:raise RuntimeError(f'{model} failed (exit {rc}). Log: {log}. Retry is safe after fixing the cause.')
    if model not in load_results(job)[1]:raise RuntimeError('No completed output found; inspect log')
    print('Completed. Continue to the comparison plots.')
