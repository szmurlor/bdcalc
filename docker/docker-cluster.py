import ray
import time
import json

@ray.remote
def getnode():
    time.sleep(0.5)
    #return ray.services.get_node_ip_address()
    return 1

ray.init(address="ray-1.iem.pw.edu.pl:6379")
#print("Connected")

#exit(1)

#ray.init(redis_address="172.17.0.2:64940")
print(json.dumps(ray.nodes(), indent=4, sort_keys=True))
print(f"Liczba wezlow: {len(ray.nodes())}")
id = getnode.remote();
print(id)
print(ray.get(id))
print(set(ray.get([getnode.remote() for _ in range(500)])))
print(ray.state.available_resources())
print(ray.state.cluster_resources())
#print(ray.state.errors())
#print(ray.state.objects())
#print(ray.state.tasks())
