"""Original cloud circuit for the operating-condition experiment."""
import socket,time,json,csv,sys,os
from pathlib import Path
import numpy as np
import tenseal as ts
from run_twohost_experiments import send,recv
PORT=47843

def server():
 with socket.socket() as listener:
  listener.bind(('0.0.0.0',PORT));listener.listen(1);print('READY',flush=True)
  with listener.accept()[0] as s:
   s.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1);init,_=recv(s)
   ctx=ts.context_from(init['context'],n_threads=1);assert not ctx.has_secret_key();coef=init['coeff'];send(s,dict(secret=False,threads=1))
   from fuzzy_controller import FuzzyLongitudinalController
   c=object.__new__(FuzzyLongitudinalController)
   while True:
    req,_=recv(s)
    if req is None:break
    if req['mode']=='echo':send(s,dict(payload=req['payload'],eval_ms=0,decode_ms=0,serialize_ms=0));continue
    t=time.perf_counter();x=[ts.ckks_vector_from(ctx,b) for b in req['blobs']];decode=(time.perf_counter()-t)*1000;t=time.perf_counter();co=coef[req['region']]
    if req['mode']=='single':
     a=x[0]
     weights=[co[4]-(co[5]+co[6])/2,co[7]-(co[5]+co[8])/2,co[9]-(co[6]+co[8])/2,co[5]/2,co[6]/2,co[8]/2]
     out=((a*a)*weights+a*[co[1],co[2],co[3],0,0,0]).sum()+co[0]
    else:
     a,b=x;weights=[co[4],co[7],co[9],co[5],co[6],co[8]]
     out=((a*b)*weights+a*[co[1],co[2],co[3],0,0,0]).sum()+co[0]
    ev=(time.perf_counter()-t)*1000;t=time.perf_counter();blob=out.serialize();ser=(time.perf_counter()-t)*1000
    send(s,dict(output=blob,eval_ms=ev,decode_ms=decode,serialize_ms=ser))

if __name__=="__main__":
 server()
