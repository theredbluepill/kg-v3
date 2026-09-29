import os, resource, threading, time

def watch_rss():
    while True:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if peak > 950_000_000:
            os.write(2, f'R4 RSS safety stop: {peak} bytes\n'.encode())
            os._exit(86)
        time.sleep(.05)
threading.Thread(target=watch_rss, daemon=True).start()
