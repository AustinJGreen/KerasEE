mkdir eedream
xcopy src\blocks.py eedream\
xcopy src\utils.py eedream\
xcopy src\dream.py eedream\
xcopy res\blocks\* eedream\res\blocks\
xcopy res\block_colors.txt eedream\res\
xcopy res\blocks_optimized.txt eedream\res\
copy models\pro_classifier\ver38\models\latest.h5 eedream\model.h5
cd eedream
echo python dream.py --world myworld.eelvl >> do_dream.cmd
echo Step 1. Download eelvl from EE. >> README.txt
echo Step 2. Put file in same directory as dream.py >> README.txt
echo Step 3. Edit do_dream.cmd and replace myworld.eelvl with the name of your level >> README.txt
echo Or, you can run python dream.py --world myworld.eelvl >> README.txt
echo *** Make sure you have python 3.6 installed. *** >> README.txt
echo: >> README.txt
echo Pro: Step 4. Open up dream.py and play with parameters. Change params (which change settings) for visual effects >> README.txt
echo and play around with step, num_octave, octave_scale, iterations and max_loss. >> README.txt
echo: >> README.txt
echo - ugp >> README.txt