# script to generate calls.
# run with `python setup_exp_commands.py > run_experiments.sh`

MODELS = ['ARMA', 'IFNN', 'IRNN', 'CG', 'JFNN', 'JRNN', 'MOGP', 'CGMM', 'CANF', 
    'EncDec', 'ResNN','QResPinb', 'QRNNPinb', 'QRNNDecPinb']
DATASETS=['openei -loc 1', 'openei -loc 9', 'iso-ne1', 'iso-ne3', 'iso-ne4', 'iso-ne2',  'nau', ]
TT = ['--train -nseeds 3', '-nseeds 10']

for d in DATASETS:
    for tt in TT:
        for inp in [8,24]:
            print(f'# {d} input={inp} training') if tt==TT[0] else print(f'# {d} input={inp} testing') 
            for m in MODELS:
                print(f'python -m experiments.test_models --ray -model {m} -dataset {d} -input {inp} {tt} -cpus_per_trial 5')
            print()
        print()
    print()

