import sciris as sc

def foo():
    print("hi there")

def bar(x):
    print(x)

if __name__ == "__main__":
    sc.parallelize(foo, iterarg=5)
    sc.parallelize(bar, iterarg=[1,2,3,4])
