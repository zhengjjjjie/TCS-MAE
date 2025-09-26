
python train_unet_ae.py -encoder=mit_b2 -input_size=256 -intensity_mask_size=6 -intensity_mask_ratio=0.75 -save_dir=mit_b2_s06_m75_intensity -epochs=45 -steps_per_epoch=2000 -lr=0.0001
python train_unet_ae.py -encoder=mit_b2 -input_size=256 -intensity_mask_size=6 -intensity_mask_ratio=0.15 -save_dir=mit_b2_s06_m15_intensity -epochs=45 -steps_per_epoch=2000 -lr=0.0001
python train_unet_ae.py -encoder=mit_b2 -input_size=256 -intensity_mask_size=6 -intensity_mask_ratio=0.35 -save_dir=mit_b2_s06_m35_intensity -epochs=45 -steps_per_epoch=2000 -lr=0.0001
python train_unet_ae.py -encoder=mit_b2 -input_size=256 -intensity_mask_size=6 -intensity_mask_ratio=0.55 -save_dir=mit_b2_s06_m55_intensity -epochs=45 -steps_per_epoch=2000 -lr=0.0001

#python train_unet_ae.py -encoder=mit_b2 -input_size=256 -intensity_mask_size=4 -intensity_mask_ratio=0.15 -save_dir=mit_b2_s04_m15_intensity -epochs=60 -steps_per_epoch=2000 -lr=0.0001
#python train_unet_ae.py -encoder=mit_b2 -input_size=256 -intensity_mask_size=4 -intensity_mask_ratio=0.35 -save_dir=mit_b2_s04_m35_intensity -epochs=60 -steps_per_epoch=2000 -lr=0.0001
#python train_unet_ae.py -encoder=mit_b2 -input_size=256 -intensity_mask_size=4 -intensity_mask_ratio=0.55 -save_dir=mit_b2_s04_m55_intensity -epochs=60 -steps_per_epoch=2000 -lr=0.0001
#python train_unet_ae.py -encoder=mit_b2 -input_size=256 -intensity_mask_size=4 -intensity_mask_ratio=0.75 -save_dir=mit_b2_s04_m75_intensity -epochs=60 -steps_per_epoch=2000 -lr=0.0001


#python train_unet_ae.py -encoder=mit_b2 -input_size=256 -intensity_mask_size=8 -intensity_mask_ratio=0.15 -save_dir=mit_b2_s08_m15_intensity -epochs=60 -steps_per_epoch=2000 -lr=0.0001
#python train_unet_ae.py -encoder=mit_b2 -input_size=256 -intensity_mask_size=8 -intensity_mask_ratio=0.35 -save_dir=mit_b2_s08_m35_intensity -epochs=60 -steps_per_epoch=2000 -lr=0.0001
#python train_unet_ae.py -encoder=mit_b2 -input_size=256 -intensity_mask_size=8 -intensity_mask_ratio=0.55 -save_dir=mit_b2_s08_m55_intensity -epochs=60 -steps_per_epoch=2000 -lr=0.0001
#python train_unet_ae.py -encoder=mit_b2 -input_size=256 -intensity_mask_size=8 -intensity_mask_ratio=0.75 -save_dir=mit_b2_s08_m75_intensity -epochs=60 -steps_per_epoch=2000 -lr=0.0001
