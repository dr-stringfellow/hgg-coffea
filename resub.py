from distributed import Client
import sys

client = Client("18.4.112.212:%s" % sys.argv[1])

if len(sys.argv) == 2:
    address = client.dashboard_link.replace("status","info/main/workers.html")
    print("webpage=%s" % address)

if len(sys.argv) == 3:
    with open(sys.argv[2]) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
        print(lines)
        client.retire_workers(lines)

