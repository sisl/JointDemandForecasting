# script to generate calls.
# run with `python setup_exp_commands.py > run_experiments.sh`

MODELS = ['ARMA', 'IFNN', 'IRNN', 'CG', 'JFNN', 'JRNN', 'CGMM', 'CANF', 
    'EncDec', 'ResNN','QResPinb', 'QRNNPinb', 'QRNNDecPinb'] # + ['MOGP']
DATASETS_CORES = [
    ('openei -loc 1', 1), 
    ('openei -loc 9', 1), 
    ('iso-ne1', 1), 
    ('iso-ne4', 4), 
    ('iso-ne2', 2), 
    ('iso-ne3', 3),  
    ('nau', 5), 
    ]
TT = ['--train -nseeds 3', '-nseeds 10']

for d, c in DATASETS_CORES:
    for tt in TT:
        for inp in [8,24]:
            print(f'# {d} input={inp} training') if tt==TT[0] else print(f'# {d} input={inp} testing') 
            for m in MODELS:
                print(f'python -m experiments.test_models --ray -model {m} -dataset {d} -input {inp} {tt} -cpus_per_trial {c}')
            print()
        print()
    print()

