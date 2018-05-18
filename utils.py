import datetime

def logmsg(msg=''):
    fmt = "%H:%M:%S"
    s = f'{datetime.datetime.today().strftime(fmt)}: {msg}'
    with open('log.txt', "a") as myfile:
        myfile.write(f'{s}\n')
     print(s)
