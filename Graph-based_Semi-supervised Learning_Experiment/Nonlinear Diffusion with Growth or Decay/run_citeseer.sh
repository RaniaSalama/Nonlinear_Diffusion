echo "Nonlinear diffusion using tanh function\n" >> citeseer_result.txt
python train.py --dataset=citeseer --nclass=7 --knighbors_number=200 --rank_based=1 --t=450 --generate_knn=1 --use_two_diffusions=1 --sigma=1.9 --w=0.5 --function="tanh" --self_learning=0 >> citeseer_result.txt
echo "Nonlinear diffusion using tanh function with self learning\n" >> citeseer_result.txt
python train.py --dataset=citeseer --nclass=7 --knighbors_number=200 --rank_based=1 --t=450 --generate_knn=0 --use_two_diffusions=1 --sigma=1.9 --w=0.5 --function="tanh" --self_learning=1 >> citeseer_result.txt
echo "Nonlinear diffusion using power function\n" >> citeseer_result.txt
python train.py --dataset=citeseer --nclass=7 --knighbors_number=200 --rank_based=1 --t=400 --p1=0.55 --p2=0.55 --generate_knn=0 --use_two_diffusions=1 --sigma=0.51 --function="power" --self_learning=0 >> citeseer_result.txt
echo "Nonlinear diffusion using power function with self learning\n" >> citeseer_result.txt
python train.py --dataset=citeseer --nclass=7 --knighbors_number=200 --rank_based=1 --t=400 --p1=0.55 --p2=0.55 --generate_knn=0 --use_two_diffusions=1 --sigma=0.51 --function="power" --self_learning=1 >> citeseer_result.txt

