bash tools/dist_train.sh projects/configs/StreamPETR/stream_petr_r50_flash_704_bs2_seq_428q_nui_60e.py 1


bash tools/dist_train.sh projects/PETR/configs/petr_vovnet_gridmask_p4_800x320.py 1

bash tools/dist_test.sh projects/PETR/configs/petr_vovnet_gridmask_p4_800x320.py /data/l50062047/working/SlarmDet/work_dirs/petr_vovnet_gridmask_p4_800x320/epoch_24.pth 8

tools/dist_train.sh projects/configs/StreamPETR/stream_petr_r50_flash_704_bs2_seq_428q_nui_60e.py 1 --work-dir work_dirs/stream_petr_r50_flash_704_bs2_seq_24e/


python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag v1.0-mini