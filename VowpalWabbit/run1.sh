vw trainSep.vw -f avazu.model.vw --loss_function logistic -q :K -q :J --l2 0.00000001  -b22  --l1 0.000000003375  -l 0.05  --passes 10 -c -k --holdout_after 32377422 



vw testSep.vw -t -i avazu.model.vw -p avazu.preds.txt -b22 -q :K -q :J --l2 0.000000001 -l 0.050 --l1 0.000000003375
