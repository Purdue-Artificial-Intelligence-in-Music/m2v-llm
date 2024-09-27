import numpy
import time

t = time.time()

for i in range(50):
    matrix = numpy.random.randn(200,200)
    numpy.linalg.inv(matrix)
    numpy.linalg.eig(matrix)

print(f"Time taken: %.3f seconds" % (time.time() - t))