import ray
from dicomutils import info


def sumator(x, y):
    return x*y


@ray.remote
def ala(x, y):
    info("ala %f" % (sumator(x, y)))
    return x+y


ray.init()

idx = [ala.remote(x, x) for x in range(100)]
print(ray.get(idx))
