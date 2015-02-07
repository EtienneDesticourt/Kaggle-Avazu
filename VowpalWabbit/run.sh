#vw temp.vw -c  --loss_function logistic --passes 1 -l 1 --l1 0 -k  --holdout_after 1000 -f avazu.model.vw -b 22


vw trainSep.vw -f avazu.model.vw --loss_function logistic --invert_hash readable.model -b22 --l1 0.000000003375  -l 0.05  --passes 1 --holdout_after 32377422
#vw trainSep.vw -f avazu.model.vw --loss_function logistic 
#vw test.vw -t -i avazu.model.vw -p avazu.preds.txt -b22  -l 0.050 --l1 0.000000003375
#roaming sheep
