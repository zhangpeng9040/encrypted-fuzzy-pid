"""Matched three-controller two-host full-course campaign; trusted LAN only."""
import socket,time,threading,queue,csv,json,hashlib,sys,os
from pathlib import Path
from collections import deque
import numpy as np
import tenseal as ts
from run_twohost_experiments import send,recv
from benchmark_request_payload import build_request,validate_request,send_checked_request
from fuzzy_controller import FuzzyLongitudinalController
PORT=47856
O=Path('original_four_twohost_payload_fixed_verified')
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
   from tase2026_fuzzy_controller import _Tase2026Longitudinal
   chen=_Tase2026Longitudinal();p=init['chen_parameters'];chen.Kp0,chen.Ki0,chen.Kd0=map(float,p[:3]);chen.e_qf=2/16;chen.ec_qf=2/3
   chen.delta_kp_list=[x*p[3] for x in chen.delta_kp_list];chen.delta_ki_list=[x*p[4] for x in chen.delta_ki_list];chen.delta_kd_list=[x*p[5] for x in chen.delta_kd_list];first=True
   request_count={};schema_by_mode={}
   send(sock,dict(secret=False))
   while True:
    req,_=recv(sock)
    if req is None:
     print(json.dumps(dict(payload_audit='passed',request_counts=request_count,request_fields=schema_by_mode)),flush=True);break
    validated_mode=validate_request(req);request_count[validated_mode]=request_count.get(validated_mode,0)+1;schema_by_mode[validated_mode]=sorted(req)
    started=time.perf_counter();co=coef[req['region']];mode=req['mode']
    if mode=='chen':
     if first:chen.prev_error=req['error'];first=False
     out=float(chen.compute_throttle(req['error']))
    elif mode=='fuzzy':z=req['z'];out=float(c._pid_throttle(z[0],z[1],ie_sum=z[2]))
    elif mode=='poly':z=req['z'];out=float(np.clip(c.polynomial_features(*z)@co,-1,1))
    else:
     a=ts.ckks_vector_from(ctx,req['blob']);w=[co[4]-(co[5]+co[6])/2,co[7]-(co[5]+co[8])/2,co[9]-(co[6]+co[8])/2,co[5]/2,co[6]/2,co[8]/2];out=(((a*a)*w+a*[co[1],co[2],co[3],0,0,0]).sum()+co[0]).serialize()
    send(sock,dict(output=out,server_ms=(time.perf_counter()-started)*1000))
def client():
 from main import load_waypoints
 from lateral_control import steering_radians
 O.mkdir(exist_ok=False)
 if (O/'complete.json').exists():raise RuntimeError('completed result directory already exists; select a fresh output directory')
 c=FuzzyLongitudinalController(False);coeff=np.load('revision_results/piecewise_coefficients.npy');_,wp,dist,dense,lookup=load_waypoints('racetrack_waypoints.txt');dense=np.array(dense)
 context=ts.context(ts.SCHEME_TYPE.CKKS,8192,coeff_mod_bit_sizes=[60,40,40,60],n_threads=1);context.global_scale=2**40;context.generate_galois_keys()
 fields=['e_mf_params','de_mf_params','u_mf_params','rule_base','u_centroids','Kp_base','Ki_base','Kd_base','e_range','de_range','ie_range']
 protocol=dict(period=.1,max_steps=3000,route='racetrack_waypoints.txt',gains=[1,.3,.001],alpha=[.5,.3,.2],coeff_sha256=hashlib.sha256(Path('revision_results/piecewise_coefficients.npy').read_bytes()).hexdigest(),route_sha256=hashlib.sha256(Path('racetrack_waypoints.txt').read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='all four control laws remote on same LAN; original corrected target384 regions; real wall-clock sampled updates; arrivals applied next sampled boundary; hold previous if pending; full course; startup included; no added noise',timing='total request includes preprocessing encryption serialization communication decoding/decryption; handshake excluded')
 protocol.update(chen_parameters=[1.,.3,.001,1.,1.,1.],chen_scale=[2/16,2/3],scope='four original-parameter controllers; same server and original course;100ms next sampling boundary; no retuning; strict per-mode request fields',chen_integral='server20sample window; raw error; first derivative zero',chen_coverage_fix='quantize physical ranges to centers[-2,2] using2/16 and2/3; original numeric gains and correction centers retained');protocol.update(payload_schema={'chen':['mode','region','error'],'fuzzy':['mode','region','z'],'poly':['mode','region','z'],'ckks_poly':['mode','region','blob']},payload_check='validate exact serialized bytes before send and deserialized request at server',payload_validator_sha256=hashlib.sha256(Path('benchmark_request_payload.py').read_bytes()).hexdigest());(O/'protocol.json').write_text(json.dumps(protocol,indent=2));summ=[];requests=[]
 with socket.create_connection((os.environ.get('FUZZY_SERVER_HOST','127.0.0.1'),PORT),timeout=30) as sock:
  sock.settimeout(120);sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1);send(sock,dict(context=context.serialize(save_secret_key=False),coeff=coeff,inference={k:getattr(c,k) for k in fields},chen_parameters=[1.,.3,.001,1.,1.,1.]));ack,_=recv(sock);assert not ack['secret']
  for mode in ['chen','fuzzy','poly','ckks_poly']:
   jobs=queue.Queue();done=queue.Queue();errors=queue.Queue()
   def worker():
    while True:
     task=jobs.get()
     if task is None:return
     k,z,raw_error=task
     try:
      start=time.perf_counter()
      def encrypt(z):
       e,d,i=map(float,z);return ts.ckks_vector(context,[e,d,i,e+d,e+i,d+i]).serialize()
      req=build_request(mode,c._region_index(*z),z,raw_error,encrypt)
      up,payload_fields=send_checked_request(sock,req);res,down=recv(sock);u=ts.ckks_vector_from(context,res['output']).decrypt()[0] if mode=='ckks_poly' else res['output'];arrival=time.perf_counter();ms=(arrival-start)*1000
      requests.append(dict(mode=mode,seq=k,payload_fields=payload_fields,payload_validated=1,total_ms=ms,server_ms=res['server_ms'],up_bytes=up,down_bytes=down,arrival=arrival));done.put((k,float(np.clip(u,-1,1)),arrival))
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
    sub=dense[lookup[first]:lookup[end]+1];near=int(np.argmin(np.sum((sub[:,:2]-[x,y])**2,axis=1)));ref=float(sub[near,2]);e=ref-v;d=(e-prev)/.1 if k else 0;prev=e;buf.append(e);z=np.clip([e,d,sum(buf)*.1],[-15,-3,-20],[15,3,20]);sent=0
    if not busy:jobs.put((k,z,e));busy=True;sent=1
    trace.append(dict(step=k,x=x,y=y,speed=v,ref=ref,error=e,command=u,applied_seq=last,request_sent=sent,scheduling_lateness_ms=max(0,boundary-target)*1000))
    steer=steering_radians(x,y,yaw,v,sub);x+=v*np.cos(yaw)*.1;y+=v*np.sin(yaw)*.1;yaw+=v/2.9*np.tan(steer)*.1;v=max(0,v+.3*u)
    if k%100==0:save(mode+'_trajectory.csv',trace);print(mode,k,flush=True)
    if progress>=len(wp)-5 and np.linalg.norm(wp[-1,:2]-[x,y])<4:completed=True;break
   jobs.put(None);thread.join(120)
   if thread.is_alive() or not errors.empty():raise RuntimeError('request worker failed')
   save(mode+'_trajectory.csv',trace);save('requests.csv',requests);err=np.array([r['error'] for r in trace]);lat=np.array([r['total_ms'] for r in requests if r['mode']==mode]);summ.append(dict(controller=mode,steps=len(trace),completed=completed,RMSE=float(np.sqrt(np.mean(err*err))),MAE=float(np.mean(abs(err))),Max_Error=float(max(abs(err))),Overshoot=float(max(0,-min(err))),mean_ms=float(lat.mean()),p95_ms=float(np.percentile(lat,95)),p99_ms=float(np.percentile(lat,99)),max_ms=float(max(lat)),over100=int(sum(lat>100)),requests=len(lat)));save('summary.csv',summ)
  send(sock,None)
 (O/'complete.json').write_text(json.dumps(dict(runs=4,all_completed=all(r['completed'] for r in summ))));print('COMPLETE',flush=True)
if __name__=='__main__':
 if '--server' in sys.argv:server()
 else:client()
