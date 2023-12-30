def derivative(func, x):
    delta = 0.0001
    return (func(x + delta) - func(x)) / delta

def function(x):
    return x*x

print(derivative(function,3))