printf "Nonlinear diffusion using tanh function\n" >> cora_result.txt
python train.py --dataset=cora --nclass=6 --knighbors_number=100 --rank_based=0 --t=450 --generate_knn=1 --use_two_diffusions=1 --sigma=1.9 --w=1.0 --function="tanh" --self_learning=0 >> cora_result.txt 
printf "Nonlinear diffusion using tanh function with self learning\n" >> cora_result.txt
python train.py --dataset=cora --nclass=6 --knighbors_number=100 --rank_based=0 --t=450 --generate_knn=0 --use_two_diffusions=1 --sigma=1.9 --w=1.0 --function="tanh" --self_learning=1 >> cora_result.txt
printf "Nonlinear diffusion using power function\n" >> cora_result.txt
python train.py --dataset=cora --nclass=6 --knighbors_number=100 --rank_based=0 --t=400 --p1=0.6 --p2=0.45 --generate_knn=0 --use_two_diffusions=1 --sigma=0.55 --function="power" --self_learning=0 >> cora_result.txt
printf "Nonlinear diffusion using power function with self learning\n" >> cora_result.txt
python train.py --dataset=cora --nclass=6 --knighbors_number=100 --rank_based=0 --t=400 --p1=0.6 --p2=0.45 --generate_knn=0 --use_two_diffusions=1 --sigma=0.55 --function="power" --self_learning=1 >> cora_result.txt

