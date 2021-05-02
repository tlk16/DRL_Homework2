echo 'Question 4'
python dqn_atari.py --memory_size=1000000 --target_type=double --model=Linear --batch_size=32 --save_name='Question4'

echo 'Question 5'
python dqn_atari.py --memory_size=1000000 --target_type=fixing --model=DQN --batch_size=32 --save_name='Question5'
