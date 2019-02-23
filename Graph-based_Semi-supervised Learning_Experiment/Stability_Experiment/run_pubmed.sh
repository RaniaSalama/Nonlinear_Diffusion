python train.py --dataset=pubmed --knighbors_number=2000 --rank_based=1 --t=300 --p1=0.7 --p2=0.6 --generate_knn=0 --use_two_diffusions=1 --sigma=0.65 --function="power" --self_learning=0
python train_change_p1.py --dataset=pubmed --knighbors_number=2000 --rank_based=1 --t=300 --p1=0.7 --p2=0.6 --generate_knn=0 --use_two_diffusions=1 --sigma=0.65 --function="power" --self_learning=0
python  train_change_p2.py --dataset=pubmed --knighbors_number=2000 --rank_based=1 --t=300 --p1=0.7 --p2=0.6 --generate_knn=0 --use_two_diffusions=1 --sigma=0.65 --function="power" --self_learning=0
python  train_change_sigma.py --dataset=pubmed --knighbors_number=2000 --rank_based=1 --t=300 --p1=0.7 --p2=0.6 --generate_knn=0 --use_two_diffusions=1 --sigma=0.65 --function="tanh" --self_learning=0

