import ray
import time
import json

@ray.remote
def getnode():
    time.sleep(0.5)
    return ray.services.get_node_ip_address()

ray.init(redis_address="ray-0:6379")
#ray.init(redis_address="172.17.0.2:64940")
print(json.dumps(ray.nodes(), indent=4, sort_keys=True))
id = getnode.remote();
print(id)
print(ray.get(id))
print(set(ray.get([getnode.remote() for _ in range(100)])))
print(ray.state.available_resources())
print(ray.state.cluster_resources())
print(ray.state.errors())
#print(ray.state.objects())
#print(ray.state.tasks())
