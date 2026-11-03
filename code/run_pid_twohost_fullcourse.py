"""Matched three-controller two-host full-course campaign; trusted LAN only."""
import socket,time,threading,queue,csv,json,hashlib,sys,os
from pathlib import Path
from collections import deque
import numpy as np
import tenseal as ts
from run_twohost_experiments import send,recv
from fuzzy_controller import FuzzyLongitudinalController
PORT=47857
O=Path('pid_twohost_fullcourse')
def save(name,rows):
 if rows:
  with (O/name).open('w',newline='',encoding='utf-8-sig') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def server():
 with socket.socket() as listener:
  listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);listener.bind(('0.0.0.0',PORT));listener.listen(1);print('READY',flush=True)
  with listener.accept()[0] as sock:
   init,_=recv(sock);ctx=ts.context_from(init['context'],n_threads=1);assert not ctx.has_secret_key();coef=np.array(init['coeff'])
   # Construct only the stateless inference attributes, without a private context or fitting.
   c=object.__new__(FuzzyLongitudinalController)
   for key,val in init['inference'].items():setattr(c,key,val)
   pid_gains=[ts.ckks_vector_from(ctx,b) for b in init['pid_gains']]
   send(sock,dict(secret=False))
   while True:
    req,_=recv(sock)
    if req is None:break
    started=time.perf_counter();co=coef[req['region']];mode=req['mode']
    if mode=='pid_plain':
     e,d,i=req['z'];out=float(np.clip(e+.3*i+.001*d,-1,1))
    elif mode=='pid_ckks':
     a=[ts.ckks_vector_from(ctx,b) for b in req['blobs']];out=(pid_gains[0]*a[0]+pid_gains[1]*a[1]+pid_gains[2]*a[2]).serialize()
    elif mode=='fuzzy':z=req['z'];out=float(c._pid_throttle(z[0],z[1],ie_sum=z[2]))
    elif mode=='poly':z=req['z'];out=float(np.clip(c.polynomial_features(*z)@co,-1,1))
    else:
     a=ts.ckks_vector_from(ctx,req['blob']);w=[co[4]-(co[5]+co[6])/2,co[7]-(co[5]+co[8])/2,co[9]-(co[6]+co[8])/2,co[5]/2,co[6]/2,co[8]/2];out=(((a*a)*w+a*[co[1],co[2],co[3],0,0,0]).sum()+co[0]).serialize()
    send(sock,dict(output=out,server_ms=(time.perf_counter()-started)*1000))
def client():
 from main import load_waypoints
 from lateral_control import steering_radians
 O.mkdir(exist_ok=True);c=FuzzyLongitudinalController(False);coeff=np.load('revision_results/piecewise_coefficients.npy');_,wp,dist,dense,lookup=load_waypoints('racetrack_waypoints.txt');dense=np.array(dense)
 context=ts.context(ts.SCHEME_TYPE.CKKS,8192,coeff_mod_bit_sizes=[60,40,40,60],n_threads=1);context.global_scale=2**40;context.generate_galois_keys()
 fields=['e_mf_params','de_mf_params','u_mf_params','rule_base','u_centroids','Kp_base','Ki_base','Kd_base','e_range','de_range','ie_range']
 protocol=dict(period=.1,max_steps=3000,route='racetrack_waypoints.txt',gains=[1,.3,.001],alpha=[.5,.3,.2],coeff_sha256=hashlib.sha256(Path('revision_results/piecewise_coefficients.npy').read_bytes()).hexdigest(),route_sha256=hashlib.sha256(Path('racetrack_waypoints.txt').read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='all three control laws remote on same LAN; original corrected target384 regions; real wall-clock sampled updates; arrivals applied next sampled boundary; hold previous if pending; full course; startup included; no added noise',timing='total request includes preprocessing encryption serialization communication decoding/decryption; handshake excluded')
 protocol.update(scope='original PID gains; plaintext PID and CKKS PID both remote; original track;100ms next sampled boundary actuation; no polynomial input clipping',derivative='raw finite difference; first sample zero',integral='20sample window; clip[-20,20]',encrypted_gains=True,ciphertext_inputs='three scalar ciphertexts[e,d,I]; linear PID baseline; not quadratic packed surrogate');protocol.pop('alpha',None);protocol.pop('coeff_sha256',None);(O/'protocol.json').write_text(json.dumps(protocol,indent=2));summ=[];requests=[]
 with socket.create_connection((os.environ.get('FUZZY_SERVER_HOST','127.0.0.1'),PORT),timeout=30) as sock:
  sock.settimeout(120);sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1);send(sock,dict(context=context.serialize(save_secret_key=False),coeff=coeff,inference={k:getattr(c,k) for k in fields},pid_gains=[ts.ckks_vector(context,[g]).serialize() for g in [1.,.3,.001]]));ack,_=recv(sock);assert not ack['secret']
  for mode in ['pid_plain']:
   jobs=queue.Queue();done=queue.Queue();errors=queue.Queue()
   def worker():
    while True:
     task=jobs.get()
     if task is None:return
     k,z=task
     try:
      start=time.perf_counter();req=dict(mode=mode,region=0)
      if mode=='pid_ckks':req['blobs']=[ts.ckks_vector(context,[float(v)]).serialize() for v in [z[0],z[2],z[1]]]
      elif mode=='ckks_poly':e,d,i=map(float,z);req['blob']=ts.ckks_vector(context,[e,d,i,e+d,e+i,d+i]).serialize()
      else:req['z']=z
      up=send(sock,req);res,down=recv(sock);u=ts.ckks_vector_from(context,res['output']).decrypt()[0] if mode in ['ckks_poly','pid_ckks'] else res['output'];arrival=time.perf_counter();ms=(arrival-start)*1000
      requests.append(dict(mode=mode,seq=k,total_ms=ms,server_ms=res['server_ms'],up_bytes=up,down_bytes=down,arrival=arrival));done.put((k,float(np.clip(u,-1,1)),arrival))
     except Exception as ex:errors.put(repr(ex));return
   thread=threading.Thread(target=worker,daemon=True);thread.start();x,y=-180.34,79.54;yaw=np.radians(20);v=u=prev=0.;buf=deque(maxlen=20);busy=False;last=-1;trace=[];progress=0;completed=False;start=time.perf_counter()
   for k in range(3000):
    target=start+k*.1;remain=target-time.perf_counter()
    if remain>0:time.sleep(remain)
    boundary=time.perf_counter()
    if not errors.empty():raise RuntimeError(errors.get())
    while not done.empty():
     seq,command,arrival=done.get();busy=False
     if seq>last:u=command;last=seq
    j=int(np.argmin(np.sum((wp[:,:2]-[x,y])**2,axis=1)));progress=max(progress,j);first=max(0,j-1);end=j;ahead=0
    while ahead<20 and end<len(wp)-1:ahead+=dist[end];end+=1
    sub=dense[lookup[first]:lookup[end]+1];near=int(np.argmin(np.sum((sub[:,:2]-[x,y])**2,axis=1)));ref=float(sub[near,2]);e=ref-v;d=(e-prev)/.1 if k else 0;prev=e;buf.append(e);z=np.array([e,d,float(np.clip(sum(buf)*.1,-20,20))]);sent=0
    if not busy:jobs.put((k,z));busy=True;sent=1
    trace.append(dict(step=k,x=x,y=y,speed=v,ref=ref,error=e,command=u,applied_seq=last,request_sent=sent,scheduling_lateness_ms=max(0,boundary-target)*1000))
    steer=steering_radians(x,y,yaw,v,sub);x+=v*np.cos(yaw)*.1;y+=v*np.sin(yaw)*.1;yaw+=v/2.9*np.tan(steer)*.1;v=max(0,v+.3*u)
    if k%100==0:save(mode+'_trajectory.csv',trace);print(mode,k,flush=True)
    if progress>=len(wp)-5 and np.linalg.norm(wp[-1,:2]-[x,y])<4:completed=True;break
   jobs.put(None);thread.join(120)
   if thread.is_alive() or not errors.empty():raise RuntimeError('request worker failed')
   save(mode+'_trajectory.csv',trace);save('requests.csv',requests);err=np.array([r['error'] for r in trace]);lat=np.array([r['total_ms'] for r in requests if r['mode']==mode]);summ.append(dict(controller=mode,steps=len(trace),completed=completed,RMSE=float(np.sqrt(np.mean(err*err))),MAE=float(np.mean(abs(err))),Max_Error=float(max(abs(err))),Overshoot=float(max(0,-min(err))),mean_ms=float(lat.mean()),p95_ms=float(np.percentile(lat,95)),p99_ms=float(np.percentile(lat,99)),max_ms=float(max(lat)),over100=int(sum(lat>100)),requests=len(lat)));save('summary.csv',summ)
  send(sock,None)
 (O/'complete.json').write_text(json.dumps(dict(runs=1,all_completed=all(r['completed'] for r in summ))));print('COMPLETE',flush=True)
if __name__=='__main__':
 if '--server' in sys.argv:server()
 else:client()
