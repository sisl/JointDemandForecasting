# openei -loc 1 input=8 training
python -m experiments.test_models --ray -model ARMA -dataset openei -loc 1 -input 8 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model IFNN -dataset openei -loc 1 -input 8 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model IRNN -dataset openei -loc 1 -input 8 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model CG -dataset openei -loc 1 -input 8 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model JFNN -dataset openei -loc 1 -input 8 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model JRNN -dataset openei -loc 1 -input 8 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model CGMM -dataset openei -loc 1 -input 8 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model CANF -dataset openei -loc 1 -input 8 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model EncDec -dataset openei -loc 1 -input 8 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model ResNN -dataset openei -loc 1 -input 8 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model QResPinb -dataset openei -loc 1 -input 8 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model QRNNPinb -dataset openei -loc 1 -input 8 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model QRNNDecPinb -dataset openei -loc 1 -input 8 --train -nseeds 3 -cpus_per_trial 1

# openei -loc 1 input=24 training
python -m experiments.test_models --ray -model ARMA -dataset openei -loc 1 -input 24 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model IFNN -dataset openei -loc 1 -input 24 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model IRNN -dataset openei -loc 1 -input 24 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model CG -dataset openei -loc 1 -input 24 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model JFNN -dataset openei -loc 1 -input 24 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model JRNN -dataset openei -loc 1 -input 24 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model CGMM -dataset openei -loc 1 -input 24 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model CANF -dataset openei -loc 1 -input 24 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model EncDec -dataset openei -loc 1 -input 24 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model ResNN -dataset openei -loc 1 -input 24 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model QResPinb -dataset openei -loc 1 -input 24 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model QRNNPinb -dataset openei -loc 1 -input 24 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model QRNNDecPinb -dataset openei -loc 1 -input 24 --train -nseeds 3 -cpus_per_trial 1


# openei -loc 1 input=8 testing
python -m experiments.test_models --ray -model ARMA -dataset openei -loc 1 -input 8 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model IFNN -dataset openei -loc 1 -input 8 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model IRNN -dataset openei -loc 1 -input 8 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model CG -dataset openei -loc 1 -input 8 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model JFNN -dataset openei -loc 1 -input 8 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model JRNN -dataset openei -loc 1 -input 8 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model CGMM -dataset openei -loc 1 -input 8 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model CANF -dataset openei -loc 1 -input 8 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model EncDec -dataset openei -loc 1 -input 8 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model ResNN -dataset openei -loc 1 -input 8 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model QResPinb -dataset openei -loc 1 -input 8 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model QRNNPinb -dataset openei -loc 1 -input 8 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model QRNNDecPinb -dataset openei -loc 1 -input 8 -nseeds 10 -cpus_per_trial 1

# openei -loc 1 input=24 testing
python -m experiments.test_models --ray -model ARMA -dataset openei -loc 1 -input 24 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model IFNN -dataset openei -loc 1 -input 24 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model IRNN -dataset openei -loc 1 -input 24 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model CG -dataset openei -loc 1 -input 24 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model JFNN -dataset openei -loc 1 -input 24 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model JRNN -dataset openei -loc 1 -input 24 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model CGMM -dataset openei -loc 1 -input 24 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model CANF -dataset openei -loc 1 -input 24 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model EncDec -dataset openei -loc 1 -input 24 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model ResNN -dataset openei -loc 1 -input 24 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model QResPinb -dataset openei -loc 1 -input 24 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model QRNNPinb -dataset openei -loc 1 -input 24 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model QRNNDecPinb -dataset openei -loc 1 -input 24 -nseeds 10 -cpus_per_trial 1



# openei -loc 9 input=8 training
python -m experiments.test_models --ray -model ARMA -dataset openei -loc 9 -input 8 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model IFNN -dataset openei -loc 9 -input 8 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model IRNN -dataset openei -loc 9 -input 8 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model CG -dataset openei -loc 9 -input 8 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model JFNN -dataset openei -loc 9 -input 8 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model JRNN -dataset openei -loc 9 -input 8 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model CGMM -dataset openei -loc 9 -input 8 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model CANF -dataset openei -loc 9 -input 8 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model EncDec -dataset openei -loc 9 -input 8 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model ResNN -dataset openei -loc 9 -input 8 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model QResPinb -dataset openei -loc 9 -input 8 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model QRNNPinb -dataset openei -loc 9 -input 8 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model QRNNDecPinb -dataset openei -loc 9 -input 8 --train -nseeds 3 -cpus_per_trial 1

# openei -loc 9 input=24 training
python -m experiments.test_models --ray -model ARMA -dataset openei -loc 9 -input 24 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model IFNN -dataset openei -loc 9 -input 24 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model IRNN -dataset openei -loc 9 -input 24 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model CG -dataset openei -loc 9 -input 24 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model JFNN -dataset openei -loc 9 -input 24 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model JRNN -dataset openei -loc 9 -input 24 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model CGMM -dataset openei -loc 9 -input 24 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model CANF -dataset openei -loc 9 -input 24 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model EncDec -dataset openei -loc 9 -input 24 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model ResNN -dataset openei -loc 9 -input 24 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model QResPinb -dataset openei -loc 9 -input 24 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model QRNNPinb -dataset openei -loc 9 -input 24 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model QRNNDecPinb -dataset openei -loc 9 -input 24 --train -nseeds 3 -cpus_per_trial 1


# openei -loc 9 input=8 testing
python -m experiments.test_models --ray -model ARMA -dataset openei -loc 9 -input 8 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model IFNN -dataset openei -loc 9 -input 8 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model IRNN -dataset openei -loc 9 -input 8 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model CG -dataset openei -loc 9 -input 8 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model JFNN -dataset openei -loc 9 -input 8 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model JRNN -dataset openei -loc 9 -input 8 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model CGMM -dataset openei -loc 9 -input 8 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model CANF -dataset openei -loc 9 -input 8 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model EncDec -dataset openei -loc 9 -input 8 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model ResNN -dataset openei -loc 9 -input 8 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model QResPinb -dataset openei -loc 9 -input 8 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model QRNNPinb -dataset openei -loc 9 -input 8 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model QRNNDecPinb -dataset openei -loc 9 -input 8 -nseeds 10 -cpus_per_trial 1

# openei -loc 9 input=24 testing
python -m experiments.test_models --ray -model ARMA -dataset openei -loc 9 -input 24 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model IFNN -dataset openei -loc 9 -input 24 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model IRNN -dataset openei -loc 9 -input 24 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model CG -dataset openei -loc 9 -input 24 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model JFNN -dataset openei -loc 9 -input 24 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model JRNN -dataset openei -loc 9 -input 24 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model CGMM -dataset openei -loc 9 -input 24 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model CANF -dataset openei -loc 9 -input 24 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model EncDec -dataset openei -loc 9 -input 24 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model ResNN -dataset openei -loc 9 -input 24 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model QResPinb -dataset openei -loc 9 -input 24 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model QRNNPinb -dataset openei -loc 9 -input 24 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model QRNNDecPinb -dataset openei -loc 9 -input 24 -nseeds 10 -cpus_per_trial 1



# iso-ne1 input=8 training
python -m experiments.test_models --ray -model ARMA -dataset iso-ne1 -input 8 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model IFNN -dataset iso-ne1 -input 8 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model IRNN -dataset iso-ne1 -input 8 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model CG -dataset iso-ne1 -input 8 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model JFNN -dataset iso-ne1 -input 8 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model JRNN -dataset iso-ne1 -input 8 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model CGMM -dataset iso-ne1 -input 8 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model CANF -dataset iso-ne1 -input 8 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model EncDec -dataset iso-ne1 -input 8 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model ResNN -dataset iso-ne1 -input 8 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model QResPinb -dataset iso-ne1 -input 8 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model QRNNPinb -dataset iso-ne1 -input 8 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model QRNNDecPinb -dataset iso-ne1 -input 8 --train -nseeds 3 -cpus_per_trial 1

# iso-ne1 input=24 training
python -m experiments.test_models --ray -model ARMA -dataset iso-ne1 -input 24 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model IFNN -dataset iso-ne1 -input 24 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model IRNN -dataset iso-ne1 -input 24 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model CG -dataset iso-ne1 -input 24 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model JFNN -dataset iso-ne1 -input 24 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model JRNN -dataset iso-ne1 -input 24 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model CGMM -dataset iso-ne1 -input 24 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model CANF -dataset iso-ne1 -input 24 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model EncDec -dataset iso-ne1 -input 24 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model ResNN -dataset iso-ne1 -input 24 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model QResPinb -dataset iso-ne1 -input 24 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model QRNNPinb -dataset iso-ne1 -input 24 --train -nseeds 3 -cpus_per_trial 1
python -m experiments.test_models --ray -model QRNNDecPinb -dataset iso-ne1 -input 24 --train -nseeds 3 -cpus_per_trial 1


# iso-ne1 input=8 testing
python -m experiments.test_models --ray -model ARMA -dataset iso-ne1 -input 8 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model IFNN -dataset iso-ne1 -input 8 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model IRNN -dataset iso-ne1 -input 8 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model CG -dataset iso-ne1 -input 8 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model JFNN -dataset iso-ne1 -input 8 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model JRNN -dataset iso-ne1 -input 8 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model CGMM -dataset iso-ne1 -input 8 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model CANF -dataset iso-ne1 -input 8 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model EncDec -dataset iso-ne1 -input 8 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model ResNN -dataset iso-ne1 -input 8 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model QResPinb -dataset iso-ne1 -input 8 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model QRNNPinb -dataset iso-ne1 -input 8 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model QRNNDecPinb -dataset iso-ne1 -input 8 -nseeds 10 -cpus_per_trial 1

# iso-ne1 input=24 testing
python -m experiments.test_models --ray -model ARMA -dataset iso-ne1 -input 24 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model IFNN -dataset iso-ne1 -input 24 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model IRNN -dataset iso-ne1 -input 24 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model CG -dataset iso-ne1 -input 24 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model JFNN -dataset iso-ne1 -input 24 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model JRNN -dataset iso-ne1 -input 24 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model CGMM -dataset iso-ne1 -input 24 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model CANF -dataset iso-ne1 -input 24 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model EncDec -dataset iso-ne1 -input 24 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model ResNN -dataset iso-ne1 -input 24 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model QResPinb -dataset iso-ne1 -input 24 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model QRNNPinb -dataset iso-ne1 -input 24 -nseeds 10 -cpus_per_trial 1
python -m experiments.test_models --ray -model QRNNDecPinb -dataset iso-ne1 -input 24 -nseeds 10 -cpus_per_trial 1



# iso-ne4 input=8 training
python -m experiments.test_models --ray -model ARMA -dataset iso-ne4 -input 8 --train -nseeds 3 -cpus_per_trial 4
python -m experiments.test_models --ray -model IFNN -dataset iso-ne4 -input 8 --train -nseeds 3 -cpus_per_trial 4
python -m experiments.test_models --ray -model IRNN -dataset iso-ne4 -input 8 --train -nseeds 3 -cpus_per_trial 4
python -m experiments.test_models --ray -model CG -dataset iso-ne4 -input 8 --train -nseeds 3 -cpus_per_trial 4
python -m experiments.test_models --ray -model JFNN -dataset iso-ne4 -input 8 --train -nseeds 3 -cpus_per_trial 4
python -m experiments.test_models --ray -model JRNN -dataset iso-ne4 -input 8 --train -nseeds 3 -cpus_per_trial 4
python -m experiments.test_models --ray -model CGMM -dataset iso-ne4 -input 8 --train -nseeds 3 -cpus_per_trial 4
python -m experiments.test_models --ray -model CANF -dataset iso-ne4 -input 8 --train -nseeds 3 -cpus_per_trial 4
python -m experiments.test_models --ray -model EncDec -dataset iso-ne4 -input 8 --train -nseeds 3 -cpus_per_trial 4
python -m experiments.test_models --ray -model ResNN -dataset iso-ne4 -input 8 --train -nseeds 3 -cpus_per_trial 4
python -m experiments.test_models --ray -model QResPinb -dataset iso-ne4 -input 8 --train -nseeds 3 -cpus_per_trial 4
python -m experiments.test_models --ray -model QRNNPinb -dataset iso-ne4 -input 8 --train -nseeds 3 -cpus_per_trial 4
python -m experiments.test_models --ray -model QRNNDecPinb -dataset iso-ne4 -input 8 --train -nseeds 3 -cpus_per_trial 4

# iso-ne4 input=24 training
python -m experiments.test_models --ray -model ARMA -dataset iso-ne4 -input 24 --train -nseeds 3 -cpus_per_trial 4
python -m experiments.test_models --ray -model IFNN -dataset iso-ne4 -input 24 --train -nseeds 3 -cpus_per_trial 4
python -m experiments.test_models --ray -model IRNN -dataset iso-ne4 -input 24 --train -nseeds 3 -cpus_per_trial 4
python -m experiments.test_models --ray -model CG -dataset iso-ne4 -input 24 --train -nseeds 3 -cpus_per_trial 4
python -m experiments.test_models --ray -model JFNN -dataset iso-ne4 -input 24 --train -nseeds 3 -cpus_per_trial 4
python -m experiments.test_models --ray -model JRNN -dataset iso-ne4 -input 24 --train -nseeds 3 -cpus_per_trial 4
python -m experiments.test_models --ray -model CGMM -dataset iso-ne4 -input 24 --train -nseeds 3 -cpus_per_trial 4
python -m experiments.test_models --ray -model CANF -dataset iso-ne4 -input 24 --train -nseeds 3 -cpus_per_trial 4
python -m experiments.test_models --ray -model EncDec -dataset iso-ne4 -input 24 --train -nseeds 3 -cpus_per_trial 4
python -m experiments.test_models --ray -model ResNN -dataset iso-ne4 -input 24 --train -nseeds 3 -cpus_per_trial 4
python -m experiments.test_models --ray -model QResPinb -dataset iso-ne4 -input 24 --train -nseeds 3 -cpus_per_trial 4
python -m experiments.test_models --ray -model QRNNPinb -dataset iso-ne4 -input 24 --train -nseeds 3 -cpus_per_trial 4
python -m experiments.test_models --ray -model QRNNDecPinb -dataset iso-ne4 -input 24 --train -nseeds 3 -cpus_per_trial 4


# iso-ne4 input=8 testing
python -m experiments.test_models --ray -model ARMA -dataset iso-ne4 -input 8 -nseeds 10 -cpus_per_trial 4
python -m experiments.test_models --ray -model IFNN -dataset iso-ne4 -input 8 -nseeds 10 -cpus_per_trial 4
python -m experiments.test_models --ray -model IRNN -dataset iso-ne4 -input 8 -nseeds 10 -cpus_per_trial 4
python -m experiments.test_models --ray -model CG -dataset iso-ne4 -input 8 -nseeds 10 -cpus_per_trial 4
python -m experiments.test_models --ray -model JFNN -dataset iso-ne4 -input 8 -nseeds 10 -cpus_per_trial 4
python -m experiments.test_models --ray -model JRNN -dataset iso-ne4 -input 8 -nseeds 10 -cpus_per_trial 4
python -m experiments.test_models --ray -model CGMM -dataset iso-ne4 -input 8 -nseeds 10 -cpus_per_trial 4
python -m experiments.test_models --ray -model CANF -dataset iso-ne4 -input 8 -nseeds 10 -cpus_per_trial 4
python -m experiments.test_models --ray -model EncDec -dataset iso-ne4 -input 8 -nseeds 10 -cpus_per_trial 4
python -m experiments.test_models --ray -model ResNN -dataset iso-ne4 -input 8 -nseeds 10 -cpus_per_trial 4
python -m experiments.test_models --ray -model QResPinb -dataset iso-ne4 -input 8 -nseeds 10 -cpus_per_trial 4
python -m experiments.test_models --ray -model QRNNPinb -dataset iso-ne4 -input 8 -nseeds 10 -cpus_per_trial 4
python -m experiments.test_models --ray -model QRNNDecPinb -dataset iso-ne4 -input 8 -nseeds 10 -cpus_per_trial 4

# iso-ne4 input=24 testing
python -m experiments.test_models --ray -model ARMA -dataset iso-ne4 -input 24 -nseeds 10 -cpus_per_trial 4
python -m experiments.test_models --ray -model IFNN -dataset iso-ne4 -input 24 -nseeds 10 -cpus_per_trial 4
python -m experiments.test_models --ray -model IRNN -dataset iso-ne4 -input 24 -nseeds 10 -cpus_per_trial 4
python -m experiments.test_models --ray -model CG -dataset iso-ne4 -input 24 -nseeds 10 -cpus_per_trial 4
python -m experiments.test_models --ray -model JFNN -dataset iso-ne4 -input 24 -nseeds 10 -cpus_per_trial 4
python -m experiments.test_models --ray -model JRNN -dataset iso-ne4 -input 24 -nseeds 10 -cpus_per_trial 4
python -m experiments.test_models --ray -model CGMM -dataset iso-ne4 -input 24 -nseeds 10 -cpus_per_trial 4
python -m experiments.test_models --ray -model CANF -dataset iso-ne4 -input 24 -nseeds 10 -cpus_per_trial 4
python -m experiments.test_models --ray -model EncDec -dataset iso-ne4 -input 24 -nseeds 10 -cpus_per_trial 4
python -m experiments.test_models --ray -model ResNN -dataset iso-ne4 -input 24 -nseeds 10 -cpus_per_trial 4
python -m experiments.test_models --ray -model QResPinb -dataset iso-ne4 -input 24 -nseeds 10 -cpus_per_trial 4
python -m experiments.test_models --ray -model QRNNPinb -dataset iso-ne4 -input 24 -nseeds 10 -cpus_per_trial 4
python -m experiments.test_models --ray -model QRNNDecPinb -dataset iso-ne4 -input 24 -nseeds 10 -cpus_per_trial 4



# iso-ne2 input=8 training
python -m experiments.test_models --ray -model ARMA -dataset iso-ne2 -input 8 --train -nseeds 3 -cpus_per_trial 2
python -m experiments.test_models --ray -model IFNN -dataset iso-ne2 -input 8 --train -nseeds 3 -cpus_per_trial 2
python -m experiments.test_models --ray -model IRNN -dataset iso-ne2 -input 8 --train -nseeds 3 -cpus_per_trial 2
python -m experiments.test_models --ray -model CG -dataset iso-ne2 -input 8 --train -nseeds 3 -cpus_per_trial 2
python -m experiments.test_models --ray -model JFNN -dataset iso-ne2 -input 8 --train -nseeds 3 -cpus_per_trial 2
python -m experiments.test_models --ray -model JRNN -dataset iso-ne2 -input 8 --train -nseeds 3 -cpus_per_trial 2
python -m experiments.test_models --ray -model CGMM -dataset iso-ne2 -input 8 --train -nseeds 3 -cpus_per_trial 2
python -m experiments.test_models --ray -model CANF -dataset iso-ne2 -input 8 --train -nseeds 3 -cpus_per_trial 2
python -m experiments.test_models --ray -model EncDec -dataset iso-ne2 -input 8 --train -nseeds 3 -cpus_per_trial 2
python -m experiments.test_models --ray -model ResNN -dataset iso-ne2 -input 8 --train -nseeds 3 -cpus_per_trial 2
python -m experiments.test_models --ray -model QResPinb -dataset iso-ne2 -input 8 --train -nseeds 3 -cpus_per_trial 2
python -m experiments.test_models --ray -model QRNNPinb -dataset iso-ne2 -input 8 --train -nseeds 3 -cpus_per_trial 2
python -m experiments.test_models --ray -model QRNNDecPinb -dataset iso-ne2 -input 8 --train -nseeds 3 -cpus_per_trial 2

# iso-ne2 input=24 training
python -m experiments.test_models --ray -model ARMA -dataset iso-ne2 -input 24 --train -nseeds 3 -cpus_per_trial 2
python -m experiments.test_models --ray -model IFNN -dataset iso-ne2 -input 24 --train -nseeds 3 -cpus_per_trial 2
python -m experiments.test_models --ray -model IRNN -dataset iso-ne2 -input 24 --train -nseeds 3 -cpus_per_trial 2
python -m experiments.test_models --ray -model CG -dataset iso-ne2 -input 24 --train -nseeds 3 -cpus_per_trial 2
python -m experiments.test_models --ray -model JFNN -dataset iso-ne2 -input 24 --train -nseeds 3 -cpus_per_trial 2
python -m experiments.test_models --ray -model JRNN -dataset iso-ne2 -input 24 --train -nseeds 3 -cpus_per_trial 2
python -m experiments.test_models --ray -model CGMM -dataset iso-ne2 -input 24 --train -nseeds 3 -cpus_per_trial 2
python -m experiments.test_models --ray -model CANF -dataset iso-ne2 -input 24 --train -nseeds 3 -cpus_per_trial 2
python -m experiments.test_models --ray -model EncDec -dataset iso-ne2 -input 24 --train -nseeds 3 -cpus_per_trial 2
python -m experiments.test_models --ray -model ResNN -dataset iso-ne2 -input 24 --train -nseeds 3 -cpus_per_trial 2
python -m experiments.test_models --ray -model QResPinb -dataset iso-ne2 -input 24 --train -nseeds 3 -cpus_per_trial 2
python -m experiments.test_models --ray -model QRNNPinb -dataset iso-ne2 -input 24 --train -nseeds 3 -cpus_per_trial 2
python -m experiments.test_models --ray -model QRNNDecPinb -dataset iso-ne2 -input 24 --train -nseeds 3 -cpus_per_trial 2


# iso-ne2 input=8 testing
python -m experiments.test_models --ray -model ARMA -dataset iso-ne2 -input 8 -nseeds 10 -cpus_per_trial 2
python -m experiments.test_models --ray -model IFNN -dataset iso-ne2 -input 8 -nseeds 10 -cpus_per_trial 2
python -m experiments.test_models --ray -model IRNN -dataset iso-ne2 -input 8 -nseeds 10 -cpus_per_trial 2
python -m experiments.test_models --ray -model CG -dataset iso-ne2 -input 8 -nseeds 10 -cpus_per_trial 2
python -m experiments.test_models --ray -model JFNN -dataset iso-ne2 -input 8 -nseeds 10 -cpus_per_trial 2
python -m experiments.test_models --ray -model JRNN -dataset iso-ne2 -input 8 -nseeds 10 -cpus_per_trial 2
python -m experiments.test_models --ray -model CGMM -dataset iso-ne2 -input 8 -nseeds 10 -cpus_per_trial 2
python -m experiments.test_models --ray -model CANF -dataset iso-ne2 -input 8 -nseeds 10 -cpus_per_trial 2
python -m experiments.test_models --ray -model EncDec -dataset iso-ne2 -input 8 -nseeds 10 -cpus_per_trial 2
python -m experiments.test_models --ray -model ResNN -dataset iso-ne2 -input 8 -nseeds 10 -cpus_per_trial 2
python -m experiments.test_models --ray -model QResPinb -dataset iso-ne2 -input 8 -nseeds 10 -cpus_per_trial 2
python -m experiments.test_models --ray -model QRNNPinb -dataset iso-ne2 -input 8 -nseeds 10 -cpus_per_trial 2
python -m experiments.test_models --ray -model QRNNDecPinb -dataset iso-ne2 -input 8 -nseeds 10 -cpus_per_trial 2

# iso-ne2 input=24 testing
python -m experiments.test_models --ray -model ARMA -dataset iso-ne2 -input 24 -nseeds 10 -cpus_per_trial 2
python -m experiments.test_models --ray -model IFNN -dataset iso-ne2 -input 24 -nseeds 10 -cpus_per_trial 2
python -m experiments.test_models --ray -model IRNN -dataset iso-ne2 -input 24 -nseeds 10 -cpus_per_trial 2
python -m experiments.test_models --ray -model CG -dataset iso-ne2 -input 24 -nseeds 10 -cpus_per_trial 2
python -m experiments.test_models --ray -model JFNN -dataset iso-ne2 -input 24 -nseeds 10 -cpus_per_trial 2
python -m experiments.test_models --ray -model JRNN -dataset iso-ne2 -input 24 -nseeds 10 -cpus_per_trial 2
python -m experiments.test_models --ray -model CGMM -dataset iso-ne2 -input 24 -nseeds 10 -cpus_per_trial 2
python -m experiments.test_models --ray -model CANF -dataset iso-ne2 -input 24 -nseeds 10 -cpus_per_trial 2
python -m experiments.test_models --ray -model EncDec -dataset iso-ne2 -input 24 -nseeds 10 -cpus_per_trial 2
python -m experiments.test_models --ray -model ResNN -dataset iso-ne2 -input 24 -nseeds 10 -cpus_per_trial 2
python -m experiments.test_models --ray -model QResPinb -dataset iso-ne2 -input 24 -nseeds 10 -cpus_per_trial 2
python -m experiments.test_models --ray -model QRNNPinb -dataset iso-ne2 -input 24 -nseeds 10 -cpus_per_trial 2
python -m experiments.test_models --ray -model QRNNDecPinb -dataset iso-ne2 -input 24 -nseeds 10 -cpus_per_trial 2



# iso-ne3 input=8 training
python -m experiments.test_models --ray -model ARMA -dataset iso-ne3 -input 8 --train -nseeds 3 -cpus_per_trial 3
python -m experiments.test_models --ray -model IFNN -dataset iso-ne3 -input 8 --train -nseeds 3 -cpus_per_trial 3
python -m experiments.test_models --ray -model IRNN -dataset iso-ne3 -input 8 --train -nseeds 3 -cpus_per_trial 3
python -m experiments.test_models --ray -model CG -dataset iso-ne3 -input 8 --train -nseeds 3 -cpus_per_trial 3
python -m experiments.test_models --ray -model JFNN -dataset iso-ne3 -input 8 --train -nseeds 3 -cpus_per_trial 3
python -m experiments.test_models --ray -model JRNN -dataset iso-ne3 -input 8 --train -nseeds 3 -cpus_per_trial 3
python -m experiments.test_models --ray -model CGMM -dataset iso-ne3 -input 8 --train -nseeds 3 -cpus_per_trial 3
python -m experiments.test_models --ray -model CANF -dataset iso-ne3 -input 8 --train -nseeds 3 -cpus_per_trial 3
python -m experiments.test_models --ray -model EncDec -dataset iso-ne3 -input 8 --train -nseeds 3 -cpus_per_trial 3
python -m experiments.test_models --ray -model ResNN -dataset iso-ne3 -input 8 --train -nseeds 3 -cpus_per_trial 3
python -m experiments.test_models --ray -model QResPinb -dataset iso-ne3 -input 8 --train -nseeds 3 -cpus_per_trial 3
python -m experiments.test_models --ray -model QRNNPinb -dataset iso-ne3 -input 8 --train -nseeds 3 -cpus_per_trial 3
python -m experiments.test_models --ray -model QRNNDecPinb -dataset iso-ne3 -input 8 --train -nseeds 3 -cpus_per_trial 3

# iso-ne3 input=24 training
python -m experiments.test_models --ray -model ARMA -dataset iso-ne3 -input 24 --train -nseeds 3 -cpus_per_trial 3
python -m experiments.test_models --ray -model IFNN -dataset iso-ne3 -input 24 --train -nseeds 3 -cpus_per_trial 3
python -m experiments.test_models --ray -model IRNN -dataset iso-ne3 -input 24 --train -nseeds 3 -cpus_per_trial 3
python -m experiments.test_models --ray -model CG -dataset iso-ne3 -input 24 --train -nseeds 3 -cpus_per_trial 3
python -m experiments.test_models --ray -model JFNN -dataset iso-ne3 -input 24 --train -nseeds 3 -cpus_per_trial 3
python -m experiments.test_models --ray -model JRNN -dataset iso-ne3 -input 24 --train -nseeds 3 -cpus_per_trial 3
python -m experiments.test_models --ray -model CGMM -dataset iso-ne3 -input 24 --train -nseeds 3 -cpus_per_trial 3
python -m experiments.test_models --ray -model CANF -dataset iso-ne3 -input 24 --train -nseeds 3 -cpus_per_trial 3
python -m experiments.test_models --ray -model EncDec -dataset iso-ne3 -input 24 --train -nseeds 3 -cpus_per_trial 3
python -m experiments.test_models --ray -model ResNN -dataset iso-ne3 -input 24 --train -nseeds 3 -cpus_per_trial 3
python -m experiments.test_models --ray -model QResPinb -dataset iso-ne3 -input 24 --train -nseeds 3 -cpus_per_trial 3
python -m experiments.test_models --ray -model QRNNPinb -dataset iso-ne3 -input 24 --train -nseeds 3 -cpus_per_trial 3
python -m experiments.test_models --ray -model QRNNDecPinb -dataset iso-ne3 -input 24 --train -nseeds 3 -cpus_per_trial 3


# iso-ne3 input=8 testing
python -m experiments.test_models --ray -model ARMA -dataset iso-ne3 -input 8 -nseeds 10 -cpus_per_trial 3
python -m experiments.test_models --ray -model IFNN -dataset iso-ne3 -input 8 -nseeds 10 -cpus_per_trial 3
python -m experiments.test_models --ray -model IRNN -dataset iso-ne3 -input 8 -nseeds 10 -cpus_per_trial 3
python -m experiments.test_models --ray -model CG -dataset iso-ne3 -input 8 -nseeds 10 -cpus_per_trial 3
python -m experiments.test_models --ray -model JFNN -dataset iso-ne3 -input 8 -nseeds 10 -cpus_per_trial 3
python -m experiments.test_models --ray -model JRNN -dataset iso-ne3 -input 8 -nseeds 10 -cpus_per_trial 3
python -m experiments.test_models --ray -model CGMM -dataset iso-ne3 -input 8 -nseeds 10 -cpus_per_trial 3
python -m experiments.test_models --ray -model CANF -dataset iso-ne3 -input 8 -nseeds 10 -cpus_per_trial 3
python -m experiments.test_models --ray -model EncDec -dataset iso-ne3 -input 8 -nseeds 10 -cpus_per_trial 3
python -m experiments.test_models --ray -model ResNN -dataset iso-ne3 -input 8 -nseeds 10 -cpus_per_trial 3
python -m experiments.test_models --ray -model QResPinb -dataset iso-ne3 -input 8 -nseeds 10 -cpus_per_trial 3
python -m experiments.test_models --ray -model QRNNPinb -dataset iso-ne3 -input 8 -nseeds 10 -cpus_per_trial 3
python -m experiments.test_models --ray -model QRNNDecPinb -dataset iso-ne3 -input 8 -nseeds 10 -cpus_per_trial 3

# iso-ne3 input=24 testing
python -m experiments.test_models --ray -model ARMA -dataset iso-ne3 -input 24 -nseeds 10 -cpus_per_trial 3
python -m experiments.test_models --ray -model IFNN -dataset iso-ne3 -input 24 -nseeds 10 -cpus_per_trial 3
python -m experiments.test_models --ray -model IRNN -dataset iso-ne3 -input 24 -nseeds 10 -cpus_per_trial 3
python -m experiments.test_models --ray -model CG -dataset iso-ne3 -input 24 -nseeds 10 -cpus_per_trial 3
python -m experiments.test_models --ray -model JFNN -dataset iso-ne3 -input 24 -nseeds 10 -cpus_per_trial 3
python -m experiments.test_models --ray -model JRNN -dataset iso-ne3 -input 24 -nseeds 10 -cpus_per_trial 3
python -m experiments.test_models --ray -model CGMM -dataset iso-ne3 -input 24 -nseeds 10 -cpus_per_trial 3
python -m experiments.test_models --ray -model CANF -dataset iso-ne3 -input 24 -nseeds 10 -cpus_per_trial 3
python -m experiments.test_models --ray -model EncDec -dataset iso-ne3 -input 24 -nseeds 10 -cpus_per_trial 3
python -m experiments.test_models --ray -model ResNN -dataset iso-ne3 -input 24 -nseeds 10 -cpus_per_trial 3
python -m experiments.test_models --ray -model QResPinb -dataset iso-ne3 -input 24 -nseeds 10 -cpus_per_trial 3
python -m experiments.test_models --ray -model QRNNPinb -dataset iso-ne3 -input 24 -nseeds 10 -cpus_per_trial 3
python -m experiments.test_models --ray -model QRNNDecPinb -dataset iso-ne3 -input 24 -nseeds 10 -cpus_per_trial 3



# nau input=8 training
python -m experiments.test_models --ray -model ARMA -dataset nau -input 8 --train -nseeds 3 -cpus_per_trial 5
python -m experiments.test_models --ray -model IFNN -dataset nau -input 8 --train -nseeds 3 -cpus_per_trial 5
python -m experiments.test_models --ray -model IRNN -dataset nau -input 8 --train -nseeds 3 -cpus_per_trial 5
python -m experiments.test_models --ray -model CG -dataset nau -input 8 --train -nseeds 3 -cpus_per_trial 5
python -m experiments.test_models --ray -model JFNN -dataset nau -input 8 --train -nseeds 3 -cpus_per_trial 5
python -m experiments.test_models --ray -model JRNN -dataset nau -input 8 --train -nseeds 3 -cpus_per_trial 5
python -m experiments.test_models --ray -model CGMM -dataset nau -input 8 --train -nseeds 3 -cpus_per_trial 5
python -m experiments.test_models --ray -model CANF -dataset nau -input 8 --train -nseeds 3 -cpus_per_trial 5
python -m experiments.test_models --ray -model EncDec -dataset nau -input 8 --train -nseeds 3 -cpus_per_trial 5
python -m experiments.test_models --ray -model ResNN -dataset nau -input 8 --train -nseeds 3 -cpus_per_trial 5
python -m experiments.test_models --ray -model QResPinb -dataset nau -input 8 --train -nseeds 3 -cpus_per_trial 5
python -m experiments.test_models --ray -model QRNNPinb -dataset nau -input 8 --train -nseeds 3 -cpus_per_trial 5
python -m experiments.test_models --ray -model QRNNDecPinb -dataset nau -input 8 --train -nseeds 3 -cpus_per_trial 5

# nau input=24 training
python -m experiments.test_models --ray -model ARMA -dataset nau -input 24 --train -nseeds 3 -cpus_per_trial 5
python -m experiments.test_models --ray -model IFNN -dataset nau -input 24 --train -nseeds 3 -cpus_per_trial 5
python -m experiments.test_models --ray -model IRNN -dataset nau -input 24 --train -nseeds 3 -cpus_per_trial 5
python -m experiments.test_models --ray -model CG -dataset nau -input 24 --train -nseeds 3 -cpus_per_trial 5
python -m experiments.test_models --ray -model JFNN -dataset nau -input 24 --train -nseeds 3 -cpus_per_trial 5
python -m experiments.test_models --ray -model JRNN -dataset nau -input 24 --train -nseeds 3 -cpus_per_trial 5
python -m experiments.test_models --ray -model CGMM -dataset nau -input 24 --train -nseeds 3 -cpus_per_trial 5
python -m experiments.test_models --ray -model CANF -dataset nau -input 24 --train -nseeds 3 -cpus_per_trial 5
python -m experiments.test_models --ray -model EncDec -dataset nau -input 24 --train -nseeds 3 -cpus_per_trial 5
python -m experiments.test_models --ray -model ResNN -dataset nau -input 24 --train -nseeds 3 -cpus_per_trial 5
python -m experiments.test_models --ray -model QResPinb -dataset nau -input 24 --train -nseeds 3 -cpus_per_trial 5
python -m experiments.test_models --ray -model QRNNPinb -dataset nau -input 24 --train -nseeds 3 -cpus_per_trial 5
python -m experiments.test_models --ray -model QRNNDecPinb -dataset nau -input 24 --train -nseeds 3 -cpus_per_trial 5


# nau input=8 testing
python -m experiments.test_models --ray -model ARMA -dataset nau -input 8 -nseeds 10 -cpus_per_trial 5
python -m experiments.test_models --ray -model IFNN -dataset nau -input 8 -nseeds 10 -cpus_per_trial 5
python -m experiments.test_models --ray -model IRNN -dataset nau -input 8 -nseeds 10 -cpus_per_trial 5
python -m experiments.test_models --ray -model CG -dataset nau -input 8 -nseeds 10 -cpus_per_trial 5
python -m experiments.test_models --ray -model JFNN -dataset nau -input 8 -nseeds 10 -cpus_per_trial 5
python -m experiments.test_models --ray -model JRNN -dataset nau -input 8 -nseeds 10 -cpus_per_trial 5
python -m experiments.test_models --ray -model CGMM -dataset nau -input 8 -nseeds 10 -cpus_per_trial 5
python -m experiments.test_models --ray -model CANF -dataset nau -input 8 -nseeds 10 -cpus_per_trial 5
python -m experiments.test_models --ray -model EncDec -dataset nau -input 8 -nseeds 10 -cpus_per_trial 5
python -m experiments.test_models --ray -model ResNN -dataset nau -input 8 -nseeds 10 -cpus_per_trial 5
python -m experiments.test_models --ray -model QResPinb -dataset nau -input 8 -nseeds 10 -cpus_per_trial 5
python -m experiments.test_models --ray -model QRNNPinb -dataset nau -input 8 -nseeds 10 -cpus_per_trial 5
python -m experiments.test_models --ray -model QRNNDecPinb -dataset nau -input 8 -nseeds 10 -cpus_per_trial 5

# nau input=24 testing
python -m experiments.test_models --ray -model ARMA -dataset nau -input 24 -nseeds 10 -cpus_per_trial 5
python -m experiments.test_models --ray -model IFNN -dataset nau -input 24 -nseeds 10 -cpus_per_trial 5
python -m experiments.test_models --ray -model IRNN -dataset nau -input 24 -nseeds 10 -cpus_per_trial 5
python -m experiments.test_models --ray -model CG -dataset nau -input 24 -nseeds 10 -cpus_per_trial 5
python -m experiments.test_models --ray -model JFNN -dataset nau -input 24 -nseeds 10 -cpus_per_trial 5
python -m experiments.test_models --ray -model JRNN -dataset nau -input 24 -nseeds 10 -cpus_per_trial 5
python -m experiments.test_models --ray -model CGMM -dataset nau -input 24 -nseeds 10 -cpus_per_trial 5
python -m experiments.test_models --ray -model CANF -dataset nau -input 24 -nseeds 10 -cpus_per_trial 5
python -m experiments.test_models --ray -model EncDec -dataset nau -input 24 -nseeds 10 -cpus_per_trial 5
python -m experiments.test_models --ray -model ResNN -dataset nau -input 24 -nseeds 10 -cpus_per_trial 5
python -m experiments.test_models --ray -model QResPinb -dataset nau -input 24 -nseeds 10 -cpus_per_trial 5
python -m experiments.test_models --ray -model QRNNPinb -dataset nau -input 24 -nseeds 10 -cpus_per_trial 5
python -m experiments.test_models --ray -model QRNNDecPinb -dataset nau -input 24 -nseeds 10 -cpus_per_trial 5



