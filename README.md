usage of helpers function to run flowseek:
- download data and weights from the scrips given in the main flowseek README.md
- clone this flowseek_helpersc
- cd flowseek_helpersc


To getting flow from MPI-sintel data
- run ./run_comparison.sh

To get flow from your own ground truth
- run
  export PYTHONPATH=[your python path]

python /proj/ciptmp/we03cyna/fickdichwindows/process_personal_video.py \
    --video [path to your video] \
    --output-dir [your output directory] \
    --use-flowseek \
    --skip-frames 10 \
    --num-pairs 5 \
    --visualize-count 5
