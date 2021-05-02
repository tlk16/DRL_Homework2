echo 'Question 2'
python dqn_atari.py --memory_size=2 --target_type=no-fixing --model=Linear --batch_size=1 --save_name='Question2'

echo 'Question 3'
python dqn_atari.py --memory_size=1000000 --target_type=fixing --model=Linear --batch_size=32 --save_name='Question3'