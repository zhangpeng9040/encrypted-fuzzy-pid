"""Original trusted-LAN framing helpers; use only between trusted test hosts."""
import socket, struct, pickle

def send(s,obj):
 b=pickle.dumps(obj,protocol=5);s.sendall(struct.pack('!I',len(b))+b);return len(b)+4

def recv(s):
 def read(n):
  b=bytearray()
  while len(b)<n:
   x=s.recv(n-len(b))
   if not x:raise EOFError('TCP connection closed')
   b.extend(x)
  return bytes(b)
 n=struct.unpack('!I',read(4))[0]
 if n>100_000_000:raise ValueError('oversized message')
 return pickle.loads(read(n)),n+4
