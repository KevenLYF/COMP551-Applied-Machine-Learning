import numpy

def sampling(n):
    pairs = [(.2, "Movies"), (.4, "COMP-551"), (.1, "Playing"), (.3, "Studying")]
    probabilities = numpy.random.multinomial(n, list(zip(*pairs))[0])
    result = list(zip(probabilities, list(zip(*pairs))[1]))
    return result

print(sampling(100))
print(sampling(1000))