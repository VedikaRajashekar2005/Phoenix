#!/bin/bash
# Start script for HTIL backend
python ContinuousLearning_DDoS.py &
python HumanReview_DDoS.py &
wait