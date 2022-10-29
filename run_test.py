from subprocess import run


for i in range(20):
    run(['python', 'test.py', '--experiment_id', f'{i}'])
