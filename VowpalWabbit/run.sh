

vw train.vw -f avazu.model.vw --loss_function logistic -b22  --l1 0.000000003375  -l 0.050 -c -k --passes 10

vw test.vw -t -i avazu.model.vw -p avazu.preds.txt -b22  -l 0.050 --l1 0.000000003375
#roaming sheep
