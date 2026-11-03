"""Wall-clock asynchronous two-host race and endurance, no delay-isolation experiment."""
import time,socket,threading,queue,csv,json,sys,os
from pathlib import Path
from collections import deque
import numpy as np
import tenseal as ts
from run_twohost_experiments import send,recv
from fuzzy_controller import FuzzyLongitudinalController
from lateral_control import steering_radians
O=Path('final_evidence/twohost')
def save(name,rows):
 if not rows:return
 with (O/name).open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 O.mkdir(exist_ok=False,parents=True)
 ctx=ts.context(ts.SCHEME_TYPE.CKKS,8192,coeff_mod_bit_sizes=[60,40,40,60],n_threads=1);ctx.global_scale=2**40;ctx.generate_galois_keys();coef=np.load('revision_results/piecewise_coefficients.npy')
 c=object.__new__(FuzzyLongitudinalController);c.e_bins=[-10,-6,-3,0,3,6,10];c.de_bins=[-2,-1,-.5,0,.5,1,2];c.ie_bins=[-10,-5,0,5,10]
 with socket.create_connection((os.environ.get('FUZZY_SERVER_HOST','127.0.0.1'),47843),timeout=120) as sock:
  sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1);sock.settimeout(120);send(sock,dict(context=ctx.serialize(save_secret_key=False),coeff=coef));ack,_=recv(sock);assert not ack['secret']
  requests=[];seq=0
  def request(z):
   nonlocal seq
   start=time.perf_counter();e,d,i=map(float,z);a=ts.ckks_vector(ctx,[e,d,i,e+d,e+i,d+i]);send(sock,dict(mode='single',region=c._region_index(*z)+active_offset,blobs=[a.serialize()]));res,_=recv(sock);raw=ts.ckks_vector_from(ctx,res['output']).decrypt()[0];seq+=1;row=dict(seq=seq,total_ms=(time.perf_counter()-start)*1000,server_eval_ms=res['eval_ms']);requests.append(row);return float(np.clip(raw,-1,1)),row['total_ms']
  from main import load_waypoints
  _,original,dist,dense,lookup=load_waypoints('racetrack_waypoints.txt');wp=np.asarray(dense);summ=[]
  for repeat in range(3):
   for scenario,gain,noise,limit,disturbance in [('nominal',3,0,1,0),('noise',3,.05,1,0),('actuator',3,0,.6,0),('disturbance',3,0,1,-.3)]:
    for label,active_offset in [('baseline',0)]:
     runrace(scenario+'_'+label+'_r'+str(repeat),gain,noise,limit,disturbance,wp,c,request,summ)
     save('requests.csv',requests)
  send(sock,None);(O/'complete.json').write_text(json.dumps(dict(runs=12,steps=1200,scope='reported original baseline; nominal, noise, command limit and disturbance; three runs each')))

def runrace(name,gain,noise,limit,dist,wp,c,request,summ):
 jobs=queue.Queue(maxsize=1);done=queue.Queue();errors=queue.Queue();stop=object()
 def worker():
  while True:
   task=jobs.get()
   if task is stop:return
   k,z=task
   try:u,ms=request(z);done.put((k,u,ms,time.perf_counter()))
   except Exception as ex:errors.put(str(ex));return
 thread=threading.Thread(target=worker,daemon=True);thread.start();rng=np.random.default_rng(1900+int(name[-1]));x,y=-180.34,79.54;yaw=np.radians(20);v=0.;u=0.;prev=0.;buf=deque(maxlen=20);trace=[];start=time.perf_counter();lastseq=-1;busy=False
 for k in range(1200):
  target=start+k*.1;remaining=target-time.perf_counter()
  if remaining>0:time.sleep(remaining)
  now=time.perf_counter()
  if not errors.empty():raise RuntimeError(errors.get())
  while not done.empty():
   seq,command,ms,arrival=done.get();busy=False
   if seq>lastseq:
    command=float(np.clip(command,-limit,limit))
    u=float(np.clip(command,u-.05,u+.05)) if name.startswith('rate_limit') else command
    lastseq=seq
  j=int(np.argmin(np.sum((wp[:,:2]-[x,y])**2,axis=1)));ref=float(wp[j,2]);e=ref-v-rng.normal(0,noise);de=(e-prev)/.1 if k else 0;prev=e;buf.append(e);z=np.clip([e,de,sum(buf)*.1],[-15,-3,-20],[15,3,20]);sent=0
  if not busy:jobs.put((k,z));busy=True;sent=1
  steer=steering_radians(x,y,yaw,v,wp[max(0,j-100):min(len(wp),j+2001)]);trace.append(dict(e_input=float(z[0]),de_input=float(z[1]),I_input=float(z[2]),input_clipped=int(e!=z[0] or de!=z[1] or sum(buf)*.1!=z[2]),step=k,t=k*.1,scheduling_lateness_ms=max(0,now-target)*1000,x=x,y=y,speed=v,ref=ref,error=ref-v,applied=u,applied_seq=lastseq,request_sent=sent))
  x+=v*np.cos(yaw)*.1;y+=v*np.sin(yaw)*.1;yaw+=v/2.9*np.tan(steer)*.1;v=max(0,v+.1*(gain*u+(dist if 30<=k*.1<60 else 0)))
  if k%100==0:save(name+'.csv',trace);print(name,k,flush=True)
 jobs.put(stop);thread.join(timeout=120)
 if thread.is_alive():raise RuntimeError('worker failed to finish')
 if not errors.empty():raise RuntimeError(errors.get())
 save(name+'.csv',trace);err=np.array([r['error'] for r in trace]);summ.append(dict(scenario=name,rmse=float(np.sqrt(np.mean(err**2))),mae=float(np.mean(abs(err))),requests_sent=sum(r['request_sent'] for r in trace),max_scheduling_lateness_ms=max(r['scheduling_lateness_ms'] for r in trace)));save('race_summary.csv',summ)
if __name__=='__main__':main()
