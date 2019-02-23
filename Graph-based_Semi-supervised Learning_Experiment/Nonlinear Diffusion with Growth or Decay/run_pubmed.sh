echo "Nonlinear diffusion using tanh function\n" >> pubmed_result.txt
python  train.py --dataset=pubmed --knighbors_number=2000 --rank_based=1 --t=400 --generate_knn=1 --use_two_diffusions=1 --sigma=1.8 --function="tanh" --w=0.3 --self_learning=0 >> pubmed_result.txt
echo "Nonlinear diffusion using tanh function with self learning\n" >> pubmed_result.txt
python  train.py --dataset=pubmed --knighbors_number=2000 --rank_based=1 --t=400 --generate_knn=0 --use_two_diffusions=1 --sigma=1.8 --function="tanh" --w=0.3 --self_learning=1 >> pubmed_result.txt
echo "Nonlinear diffusion using power function\n" >> pubmed_result.txt
python  train.py --dataset=pubmed --knighbors_number=2000 --rank_based=1 --t=300 --p1=0.7 --p2=0.6 --generate_knn=0 --use_two_diffusions=1 --sigma=0.65 --function="power" --self_learning=0 >> pubmed_result.txt
echo "Nonlinear diffusion using power function with self learning\n" >> pubmed_result.txt
python  train.py --dataset=pubmed --knighbors_number=2000 --rank_based=1 --t=300 --p1=0.7 --p2=0.6 --generate_knn=0 --use_two_diffusions=1 --sigma=0.65 --function="power" --self_learning=1 >> pubmed_result.txt
