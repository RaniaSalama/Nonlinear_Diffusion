python train.py --dataset=citeseer --nclass=7 --knighbors_number=200 --rank_based=1 --t=400 --p1=0.55 --p2=0.55 --generate_knn=1 --use_two_diffusions=1 --sigma=0.51 --function="power" --self_learning=0
python train_change_p1.py --dataset=citeseer --nclass=7 --knighbors_number=200 --rank_based=1 --t=400 --p1=0.55 --p2=0.55 --generate_knn=1 --use_two_diffusions=1 --sigma=0.51 --function="power" --self_learning=0
python train_change_p2.py --dataset=citeseer --nclass=7 --knighbors_number=200 --rank_based=1 --t=400 --p1=0.55 --p2=0.55 --generate_knn=1 --use_two_diffusions=1 --sigma=0.51 --function="power" --self_learning=0
python train_change_sigma.py --dataset=citeseer --nclass=7 --knighbors_number=200 --rank_based=1 --t=400 --p1=0.55 --p2=0.55 --generate_knn=1 --use_two_diffusions=1 --sigma=0.51 --function="tanh" --self_learning=0

