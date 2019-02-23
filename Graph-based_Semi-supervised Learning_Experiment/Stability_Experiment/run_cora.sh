python train_change_p1.py --dataset=cora --nclass=6 --knighbors_number=100 --rank_based=0 --t=400 --p1=0.6 --p2=0.45 --generate_knn=0 --use_two_diffusions=1 --sigma=0.55 --function="power" --self_learning=0
python train_change_p2.py --dataset=cora --nclass=6 --knighbors_number=100 --rank_based=0 --t=400 --p1=0.6 --p2=0.45 --generate_knn=0 --use_two_diffusions=1 --sigma=0.55 --function="power" --self_learning=0
python train_change_sigma.py --dataset=cora --nclass=6 --knighbors_number=100 --rank_based=0 --t=400 --p1=0.6 --p2=0.45 --generate_knn=0 --use_two_diffusions=1 --sigma=0.55 --function="power" --self_learning=0
python train.py --dataset=cora --nclass=6 --knighbors_number=100 --rank_based=0 --t=400 --generate_knn=0 --use_two_diffusions=1 --sigma=0.55 --function="tanh" --self_learning=0

