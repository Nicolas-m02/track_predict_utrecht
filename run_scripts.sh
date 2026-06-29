#!/bin/bash

docker exec -d prediction_image \
    python /utrecht_exp/code/socket_experiments/prediction_module_with_gui.py

docker exec -d motion_estimation_image \
    python /utrecht_exp/segmentation/point_tracking_module.py

docker exec -d gui_image \
    uvicorn track_predict_utrecht.gui_interactive.app2:app \
    --host 0.0.0.0 \
    --port 8500 \
    --reload
    
    
    
    
    
