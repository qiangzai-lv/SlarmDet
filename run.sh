bash tools/dist_train.sh projects/configs/StreamPETR/stream_petr_r50_flash_704_bs2_seq_428q_nui_60e.py 1


bash tools/dist_train.sh projects/SLARMPETR/configs/slarm_petr_singleframe.py 1

bash tools/dist_test.sh projects/PETR/configs/petr_vovnet_gridmask_p4_800x320.py /data/working/SlarmDet/pretrain/petr_vovnet_gridmask_p4_800x320-e2191752.pth 1



python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag v1.0-mini