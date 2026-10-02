import requests

import socks
import socket
import ssl
from urllib.request import Request, urlopen
IP_ADDR = '127.0.0.1'
PORT=1080
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
socks.set_default_proxy(socks.SOCKS5, IP_ADDR, PORT)
socket.socket = socks.socksocket

def check_internet(url='http://www.google.com', timeout=5):
    try:
        response = requests.get(url, timeout=timeout)
        return True if response.status_code == 200 else False
    except (requests.ConnectionError, requests.Timeout):
        return False

def main():
    if check_internet():
        print("Internet is connected")
    else:
        print("No internet connection")