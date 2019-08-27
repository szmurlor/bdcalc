import ray
import time
import json

@ray.remote
def getnode():
    time.sleep(0.1)
    return ray.services.get_node_ip_address()
#    return ray.worker.task_context()

ray.init(redis_address="ray-0:6379")
print(json.dumps(ray.nodes(), indent=4, sort_keys=True))
print(set(ray.get([getnode.remote() for _ in range(100)])))
print(ray.state.available_resources())
print(ray.state.cluster_resources())
print(ray.state.errors())
#print(ray.state.objects())
#print(ray.state.tasks())
