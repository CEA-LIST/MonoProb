if [ "$#" -eq  "0" ]
then
    echo Usage: eval_monoprob.sh /path/to/kitti_data
    exit
fi

# MonoProb

~/miniconda3/envs/brightflow2/bin/python evaluate_depth.py \
--load_weights_folder checkpoints/S/S_640x192_resnet18_monoprob/ \
--data_path $1 \
--eval_stereo \
--eval_split eigen_benchmark \
--uncertainty \
--distribution normal \
--uncert_act sigmoid \
--uncert_as_a_fraction_of_depth \

~/miniconda3/envs/brightflow2/bin/python evaluate_depth.py \
--load_weights_folder checkpoints/S/S_640x192_resnet50_monoprob/ \
--data_path $1 \
--eval_stereo \
--eval_split eigen_benchmark \
--uncertainty \
--distribution normal \
--uncert_act sigmoid \
--uncert_as_a_fraction_of_depth \
--num_layers 50 \

~/miniconda3/envs/brightflow2/bin/python evaluate_depth.py \
--load_weights_folder checkpoints/S/S_1024x320_resnet50_monoprob/ \
--data_path $1 \
--eval_stereo \
--eval_split eigen_benchmark \
--uncertainty \
--distribution normal \
--uncert_act sigmoid \
--uncert_as_a_fraction_of_depth \
--num_layers 50 \
--width 1024 \
--height 320 \

# Self-distilled MonoProb

~/miniconda3/envs/brightflow2/bin/python evaluate_depth.py \
--load_weights_folder checkpoints/S/S_640x192_resnet18_monoprob_self/ \
--data_path $1 \
--eval_stereo \
--eval_split eigen_benchmark \
--uncertainty \
--distribution normal \
--uncert_act exp \

~/miniconda3/envs/brightflow2/bin/python evaluate_depth.py \
--load_weights_folder checkpoints/S/S_640x192_resnet50_monoprob_self/ \
--data_path $1 \
--eval_stereo \
--eval_split eigen_benchmark \
--uncertainty \
--distribution normal \
--uncert_act exp \
--num_layers 50 \

~/miniconda3/envs/brightflow2/bin/python evaluate_depth.py \
--load_weights_folder checkpoints/S/S_1024x320_resnet50_monoprob_self/ \
--data_path $1 \
--eval_stereo \
--eval_split eigen_benchmark \
--uncertainty \
--distribution normal \
--uncert_act exp \
--num_layers 50 \
--width 1024 \
--height 320 \
