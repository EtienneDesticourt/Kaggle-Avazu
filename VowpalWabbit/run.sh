

vw train.vw -f avazu.model.vw --loss_function logistic -b22  -l 0.5 -c --passes 10 --holdout_after 32377422

vw test.vw -t -i avazu.model.vw -p avazu.preds.txt -b22  -l 0.5
