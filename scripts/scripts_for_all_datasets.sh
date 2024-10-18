python main.py --name Bladder_10X --n_clusters 4 --lr 0.0001 --heads [1,1] --k 80 --fine_epochs 150 --epochs 400
python main.py --name Limb_Muscle_10X --n_clusters 6 --lr 0.0001 --heads [1,1] --k 75 --fine_epochs 150 --epochs 700
python main.py --name Trachea_10X --n_clusters 5 --lr 0.001 --heads [1,1] --k 75 --fine_epochs 100 -epochs 400
python main.py --name Adam_Drop-seq --n_clusters 8 --lr 0.0002 --heads [6,1] --k 75 --fine_epochs 100 --epochs 600 --batch_size 256
python main.py --name Muraro_CEL-seq2 --n_clusters 9 --lr 0.0001 --heads [1,1] --k 75 --fine_epochs 100 --epochs 400 --batch_size 1024
python main.py --name Plasschaert_inDrop --n_clusters 8 --lr 0.0001 --heads [1,1] --k 50 --fine_epochs 100 --epochs 400 --batch_size 256
python main.py --name Diaphragm_Smart-seq2 --n_clusters 5 --lr 0.0002 --heads [6,1] --k 75 --fine_epochs 100 --epochs 600 --batch_size 256
python main.py --name Heart_Smart-seq2 --n_clusters 8 --lr 0.0001 --heads [6,1] --k 75 --fine_epochs 100 --epochs 600 --batch_size 1024
python main.py --name Limb_Muscle_Smart-seq2 --n_clusters 6 --lr 0.0001 --heads [6,1] --k 70 --fine_epochs 100 --epochs 600 --batch_size 1024
python main.py --name Lung_Smart-seq2 --n_clusters 11 --lr 0.0001 --heads [6,1] --k 75 --fine_epochs 100 --epochs 600 --batch_size 1024
python main.py --name Trachea_Smart-seq2 --n_clusters 4 --lr 0.0002 --heads [6,1] --k 75 --fine_epochs 100 --epochs 600 --batch_size 1024
python main.py --name Young_10X --n_clusters 11 --lr 0.0001 --heads [1,1] --k 50 --fine_epochs 100 --epochs 400 --batch_size 256