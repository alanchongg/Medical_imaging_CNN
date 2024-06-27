def keylog(model_name, msg=''):
    with open('keylogs.txt', 'a') as f:
        print(f'{model_name} {msg}', file=f)

def tablelog(table):
    with open('keylogs.txt', 'a') as f:
        print(table, file=f)