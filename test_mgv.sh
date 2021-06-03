CUDA_VISIBLE_DEVICES = 5 python test_mgv.py --trained_model_folder ./selected_weights_mobilenet/ --backbone mobile0.25 \
--dataset_folder /home/sonnn27/WorkSpace/Datasets/mgv_images/Gallery \
--save_folder ./mgv_txt_evaluate/mgv_txt_mobilenet/gallery/