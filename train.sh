CUDA_VISIBLE_DEVICES=1 python train.py --training_dataset /home/sonnn27/WorkSpace/Datasets/rotated_widerface/widerface/widerface/train/label.txt \
--network mobile0.25 --num_workers 6 --lr 0.0001 --resume_net ./pretrained_model/mobilenet0.25_Final.pth