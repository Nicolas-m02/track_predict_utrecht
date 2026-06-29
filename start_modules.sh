#!/bin/bash

docker exec -d prediction_container \
    python /utrecht_exp/code/socket_experiments/prediction_module_with_gui.py

docker exec -d motion_estimation_container \
    python /utrecht_exp/segmentation/point_tracking_module.py

docker exec -d gui_container \
    uvicorn track_predict_utrecht.gui_interactive.app2:app \
    --host 0.0.0.0 \
    --port 8500 \
    --reload
