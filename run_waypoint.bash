
flag="--EXP_ID wp-train

      --TRAINEVAL train
      --VIS 0

      --ANGLES 120
      --NUM_IMGS 12

      --EPOCH 300
      --BATCH_SIZE 8
      --LEARNING_RATE 1e-6

      --WEIGHT 0

      --TRM_LAYER 2
      --TRM_NEIGHBOR 1
      --HEATMAP_OFFSET 5
      --HIDDEN_DIM 768"

python waypoint_predictor.py $flag
