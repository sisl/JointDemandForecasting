import os
import datetime
import json
import pdb
opj = os.path.join
raydir = '/home/arec/ray_results/'

all_ray_exps = [x for x in os.listdir(raydir) if os.path.isdir(opj(raydir,x)) and 'train_test' in x]

#filter times
start_time = datetime.datetime(2022, 7,15).timestamp()
end_time = datetime.datetime(2022, 8, 9).timestamp()
exps = []
for f in all_ray_exps:
    t = os.stat(opj(raydir,f)).st_ctime
    if (t > start_time) & (t < end_time):
        exps.append(f)

# get info from experiments
exps.sort()
info = {}
print(f'Total experiments: {len(exps)}')
for exp in exps:
    subs = [x for x in os.listdir(opj(raydir,exp)) if os.path.isdir(opj(raydir,exp, x))]
    num_exps = len(subs)
    error_files, metric_files = [], []
    for s in subs:
        if 'error.txt' in os.listdir(opj(raydir,exp,s)):
            error_files.append(opj(raydir,exp,s,'error.txt'))
        if 'metrics.json' in os.listdir(opj(raydir,exp,s)):
            metric_files.append(opj(raydir,exp,s,'metrics.json'))
    errors = len(error_files)
    oom = False
    for ef in error_files:
        with open(ef,'rb') as f:
            rf = str(f.read())
            if 'RayOutOfMemoryError' in rf:
                oom = True
                break

    successes = len(metric_files)
    sr = successes/num_exps
    if errors + successes != num_exps:
        pass
        #pdb.set_trace()
    param_file = opj(raydir,exp,subs[0],'params.json')
    with open(param_file,'rb') as f:
        config = json.load(f)
    try:
        model = config['model_name']
        test = config['seed'] > 90
    except:
        continue # ran into bug where a test run chose the old model config that didnt have model_name

    info[exp] = {
        'exp': exp,
        'model': model,
        'nhyps': num_exps,
        'errors': errors,
        'successes': successes,
        'success_rate': sr,
        'test':test,
        'oom':oom,
        'param_file': param_file,
        'error_files':error_files,
        'metric_files':metric_files, 
    }
    tt = 'test' if test else 'train'
    print(f'{exp}: Model {model} {tt}, {100*sr}% successes, {successes}/{errors}/{num_exps}, OoM {oom}')

pdb.set_trace()
a=0
    
# notes for debugging
# `find . -printf "%p %s %TH:%TM:%TS \n"` | grep 'error.txt'
# out.log - full log, error.txt - final stacktrace, result.json - ray result, metrics.json - saved metric
    