echo 'Question 6'
python dqn_atari.py --memory_size=1000000 --target_type=double --model=DQN --batch_size=32 --save_name='Question6'

echo 'Question 7'
python dqn_atari.py --memory_size=1000000 --target_type=fixing --model=Dueling --batch_size=32 --save_name='Question7'