echo 'Question 2'
python dqn_atari.py --memory_size=2 --target_type=no-fixing --model=Linear --batch_size=1

echo 'Question 3'
python dqn_atari.py --memory_size=1000000 --target_type=fixing --model=Linear --batch_size=32

echo 'Question 4'
python dqn_atari.py --memory_size=1000000 --target_type=double --model=Linear --batch_size=32

echo 'Question 5'
python dqn_atari.py --memory_size=1000000 --target_type=fixing --model=DQN --batch_size=32

echo 'Question 6'
python dqn_atari.py --memory_size=1000000 --target_type=double --model=DQN --batch_size=32

echo 'Question 7'
python dqn_atari.py --memory_size=1000000 --target_type=fixing --model=Dueling --batch_size=32