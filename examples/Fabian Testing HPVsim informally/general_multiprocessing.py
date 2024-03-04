import multiprocessing as mp
import os

def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())

#global que
que= mp.Queue()
def f(name, quee):
    info('function f')
    print('hello', name)
    quee.put(42)
    quee.put([2,"hello"])
    quee.put(False)
    print("finihsing f")


def g(quee):
    print("starting g")
    while True:
        x = quee.get(block=True, timeout = None)
        if x == False:
            break
        else:
            print(x)
    print("End of queue reached")


if __name__ == '__main__':
    info('main line')
    q = mp.Process(target=g, args = (que,))
    q.start()
    p = mp.Process(target=f, args=('bob',que))
    p.start()
    p.join() #join gives us a synchronisation point; it says 'i am going to wait here until all the processes are finished, then we will cotninue our execution
                #therefore, p.join() will not return until process p is complete
    q.join()
    print("program done")