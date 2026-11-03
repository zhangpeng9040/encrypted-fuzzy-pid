"""One full-course run per encrypted implementation, plus correctness preflight.

Uses the validated original plant and 100ms next-boundary feedback rule.
Trusted-LAN experiment only; pickle is not an authenticated transport.
"""
import argparse
import ast
import csv
import hashlib
import json
import os
import pickle
import platform
import queue
import socket
import struct
import sys
import threading
import time
import traceback
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import tenseal as ts
from packed_pid_ckks import PID_GAINS, SLOT_ORDER, pack_inputs, pack_gains, evaluate_packed, validate_request

MODES = ('pid_packed',)
HASH_FILES = ['run_packed_pid_twohost.py', 'packed_pid_ckks.py', 'main.py',
              'lateral_control.py', 'racetrack_waypoints.txt',
              'revision_results/piecewise_coefficients.npy']


def utc():
    return datetime.now(timezone.utc).isoformat()


def hashes():
    return {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in HASH_FILES}


def machine():
    return dict(platform=platform.platform(), processor=platform.processor(), python=sys.version,
                cpu_count=os.cpu_count(), pid=os.getpid(), numpy=np.__version__, tenseal=ts.__version__,
                load_average=list(os.getloadavg()) if hasattr(os, 'getloadavg') else None)


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def save_csv(path, rows):
    if rows:
        with Path(path).open('w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def send(sock, obj):
    body = pickle.dumps(obj, protocol=5)
    sock.sendall(struct.pack('!I', len(body)) + body)
    return len(body) + 4


def recv(sock):
    def read(n):
        result = bytearray()
        while len(result) < n:
            part = sock.recv(n-len(result))
            if not part:
                raise EOFError('TCP connection closed')
            result.extend(part)
        return bytes(result)
    length = struct.unpack('!I', read(4))[0]
    if length > 100_000_000:
        raise ValueError('oversized message')
    return pickle.loads(read(length)), length+4


def server(args):
    coefficient = np.load('revision_results/piecewise_coefficients.npy')
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(('0.0.0.0', args.port))
        listener.listen(1)
        print('READY', args.port, utc(), flush=True)
        with listener.accept()[0] as sock:
            sock.settimeout(180)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            init, _ = recv(sock)
            assert init['hashes'] == hashes(), 'source/data mismatch between hosts'
            context = ts.context_from(init['context'], n_threads=1)
            assert not context.has_secret_key()
            scalar_gains = [ts.ckks_vector_from(context, b) for b in init['scalar_gains']]
            packed_gains = ts.ckks_vector_from(context, init['packed_gains'])
            assert packed_gains.size() == 3
            send(sock, dict(secret_key=False, hashes=hashes(), machine=machine()))
            accepted = {m: 0 for m in MODES}
            rejected = 0
            while True:
                req, _ = recv(sock)
                if req is None:
                    print('FINISHED', json.dumps(accepted), 'rejected', rejected, flush=True)
                    return
                if isinstance(req, dict) and set(req) == {'command'}:
                    if req['command'] == 'snapshot':
                        send(sock, dict(accepted=accepted, rejected=rejected, machine=machine()))
                        continue
                    raise ValueError('invalid command')
                t0 = time.perf_counter()
                try:
                    mode = validate_request(req)
                except ValueError as ex:
                    rejected += 1
                    send(sock, dict(error=str(ex), rejected=True))
                    continue
                vectors = [ts.ckks_vector_from(context, b) for b in req['blobs']] if mode == 'pid_scalar' else [ts.ckks_vector_from(context, req['blob'])]
                t1 = time.perf_counter()
                if mode == 'pid_scalar':
                    out = scalar_gains[0]*vectors[0] + scalar_gains[1]*vectors[1] + scalar_gains[2]*vectors[2]
                elif mode == 'pid_packed':
                    out = evaluate_packed(vectors[0], packed_gains)
                else:
                    co = coefficient[req['region']]
                    a = vectors[0]
                    w = [co[4]-(co[5]+co[6])/2, co[7]-(co[5]+co[8])/2,
                         co[9]-(co[6]+co[8])/2, co[5]/2, co[6]/2, co[8]/2]
                    out = ((a*a)*w + a*[co[1],co[2],co[3],0,0,0]).sum()+co[0]
                t2 = time.perf_counter()
                blob = out.serialize()
                t3 = time.perf_counter()
                accepted[mode] += 1
                send(sock, dict(output=blob, payload_fields=','.join(sorted(req)),
                     server_ms=(t3-t0)*1000, server_deserialize_ms=(t1-t0)*1000,
                     server_evaluate_ms=(t2-t1)*1000, server_serialize_ms=(t3-t2)*1000))


def region(z):
    e, d, i = [int(np.searchsorted(cuts, value, side='right')) for cuts, value in
               zip(([-10,-6,-3,0,3,6,10],[-2,-1,-.5,0,.5,1,2],[-10,-5,0,5,10]),z)]
    return (e*8+d)*6+i


class Session:
    def __init__(self, sock, context):
        self.sock, self.context, self.rows = sock, context, []

    def request(self, mode, seq, z, phase):
        t0 = time.perf_counter()
        e, d, integral = map(float, z)
        req = dict(mode=mode)
        if mode == 'pid_scalar':
            enc = [ts.ckks_vector(self.context, [v]) for v in [e, integral, d]]
        elif mode == 'pid_packed':
            enc = [pack_inputs(self.context, z)]
        else:
            enc = [ts.ckks_vector(self.context, [e,d,integral,e+d,e+integral,d+integral])]
            req['region'] = region(z)
        t1 = time.perf_counter()
        blobs = [v.serialize() for v in enc]
        if mode == 'pid_scalar':
            req['blobs'] = blobs
        else:
            req['blob'] = blobs[0]
        body = pickle.dumps(req, protocol=5)
        checked = pickle.loads(body)
        validate_request(checked)
        fields = ','.join(sorted(checked))
        t2 = time.perf_counter()
        self.sock.sendall(struct.pack('!I',len(body))+body)
        response, down = recv(self.sock)
        t3 = time.perf_counter()
        assert fields == response['payload_fields']
        raw = float(ts.ckks_vector_from(self.context, response['output']).decrypt()[0])
        assert np.isfinite(raw)
        arrived = time.perf_counter()
        self.rows.append(dict(mode=mode,phase=phase,seq=seq,total_ms=(arrived-t0)*1000,
             encrypt_ms=(t1-t0)*1000, serialize_envelope_ms=(t2-t1)*1000,
             exchange_ms=(t3-t2)*1000,decrypt_ms=(arrived-t3)*1000,
             server_ms=response['server_ms'],server_deserialize_ms=response['server_deserialize_ms'],
             server_evaluate_ms=response['server_evaluate_ms'],server_serialize_ms=response['server_serialize_ms'],
             up_bytes=len(body)+4,down_bytes=down,payload_fields=fields,payload_validated=1,
             arrival=arrived,raw_output=raw))
        return float(np.clip(raw,-1,1)),arrived,raw


def load_route():
    # Use the exact validated loader without importing plotting/controller modules.
    tree = ast.parse(Path('main.py').read_text(encoding='utf-8-sig'))
    function = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='load_waypoints')
    scope = dict(np=np,csv=csv,INTERP_DISTANCE_RES=.01)
    exec(compile(ast.Module(body=[function],type_ignores=[]),'main.py','exec'),scope)
    _,wp,distance,dense,lookup = scope['load_waypoints']('racetrack_waypoints.txt')
    return wp,distance,np.array(dense),lookup


def full_course(session, mode, route, folder, args):
    from lateral_control import steering_radians
    wp,distance,dense,lookup = route
    jobs,done,errors = queue.Queue(),queue.Queue(),queue.Queue()
    def worker():
        while True:
            task = jobs.get()
            if task is None:
                return
            k,z,enqueued = task
            try:
                began = time.perf_counter()
                u,arrival,_ = session.request(mode,k,z,'formal')
                session.rows[-1].update(queue_ms=(began-enqueued)*1000,sample_to_response_ms=(arrival-enqueued)*1000)
                done.put((k,u,arrival))
            except Exception:
                errors.put(traceback.format_exc())
                return
    thread = threading.Thread(target=worker,daemon=True)
    thread.start()
    x,y,yaw,v,u,prev = -180.34,79.54,np.radians(20),0.,0.,0.
    buf = deque(maxlen=20)
    busy,last,progress,completed = False,-1,0,False
    trace = []
    started = time.perf_counter()
    try:
        for k in range(3000):
            target = started+k*.1
            remain = target-time.perf_counter()
            if remain > 0:
                time.sleep(remain)
            boundary = time.perf_counter()
            if not errors.empty():
                raise RuntimeError(errors.get())
            while not done.empty():
                seq,command,arrival = done.get()
                busy = False
                if seq > last:
                    u,last = command,seq
            j = int(np.argmin(np.sum((wp[:,:2]-[x,y])**2,axis=1)))
            progress = max(progress,j)
            first,end,ahead = max(0,j-1),j,0.
            while ahead < 20 and end < len(wp)-1:
                ahead += distance[end]
                end += 1
            sub = dense[lookup[first]:lookup[end]+1]
            near = int(np.argmin(np.sum((sub[:,:2]-[x,y])**2,axis=1)))
            ref = float(sub[near,2])
            error = ref-v
            d = (error-prev)/.1 if k else 0.
            prev = error
            buf.append(error)
            z = np.array([error,d,float(np.clip(sum(buf)*.1,-20,20))])
            if mode == 'ckks_poly':
                z = np.clip(z,[-15,-3,-20],[15,3,20])
            sent = 0
            if not busy:
                jobs.put((k,z,time.perf_counter()))
                busy,sent = True,1
            trace.append(dict(step=k,x=x,y=y,speed=v,ref=ref,error=error,command=u,
                         applied_seq=last,request_sent=sent,scheduling_lateness_ms=max(0,boundary-target)*1000))
            steer = steering_radians(x,y,yaw,v,sub)
            x += v*np.cos(yaw)*.1
            y += v*np.sin(yaw)*.1
            yaw += v/2.9*np.tan(steer)*.1
            v = max(0,v+.3*u)
            if k%300 == 0:
                write_json(args.output/'status.json',dict(state='running',controller=mode,step=k,updated_utc=utc()))
                print(mode,'step',k,flush=True)
            if progress >= len(wp)-5 and np.linalg.norm(wp[-1,:2]-[x,y]) < 4:
                completed = True
                break
    finally:
        jobs.put(None)
        thread.join(125)
        save_csv(folder/'trajectory.csv',trace)
        save_csv(folder/'requests.csv',session.rows)
    if thread.is_alive() or not errors.empty():
        raise RuntimeError('request worker failed: '+(errors.get() if not errors.empty() else 'join timeout'))
    return trace,completed


def preflight(session, coeff, args):
    rng = np.random.default_rng(428730)
    points = [[0.,0.,0.],[.2,-.7,.5],[-.3,2.,-.4],[.1,8.,-.2],[-30.,200.,20.],[30.,-200.,-20.]]
    points += rng.uniform([-2,-30,-3],[2,30,3],(26,3)).tolist()
    comparisons = []
    for k,z in enumerate(points):
        for mode in MODES[k%3:]+MODES[:k%3]:
            inp = np.clip(z,[-15,-3,-20],[15,3,20]) if mode=='ckks_poly' else z
            clipped,_,raw = session.request(mode,k,inp,'preflight')
            e,d,i = inp
            if mode=='ckks_poly':
                target = float(np.array([1,e,d,i,e*e,e*d,e*i,d*d,d*i,i*i])@coeff[region(inp)])
            else:
                target = e+.3*i+.001*d
            comparisons.append(dict(mode=mode,seq=k,expected=target,actual=raw,
                                   abs_error=abs(raw-target),clipped_abs_error=abs(clipped-np.clip(target,-1,1))))
    assert max(r['abs_error'] for r in comparisons)<1e-4
    rejection_checks = 0
    for mode in MODES:
        valid = dict(mode=mode,blobs=[b'x']*3) if mode=='pid_scalar' else dict(mode=mode,blob=b'x')
        if mode=='ckks_poly':
            valid['region']=0
        for name in ['raw_error','error','z','gains']:
            invalid = dict(valid,**{name:0.})
            try:
                validate_request(invalid)
            except ValueError:
                pass
            else:
                raise AssertionError('client accepted plaintext extra field')
            send(session.sock,invalid)
            response,_ = recv(session.sock)
            assert response['rejected'] is True
            rejection_checks += 1
    save_csv(args.output/'preflight_outputs.csv',comparisons)
    save_csv(args.output/'preflight_requests.csv',session.rows)
    report = dict(passed=True,points_per_mode=len(points),valid_requests=len(comparisons),
                  extra_field_rejections_at_both_ends=rejection_checks,
                  max_abs_error_by_mode={m:max(r['abs_error'] for r in comparisons if r['mode']==m) for m in MODES},
                  scope='Correctness samples, not a uniform error bound or independent repeated timing study.')
    write_json(args.output/'preflight.json',report)
    print('PREFLIGHT_PASSED',json.dumps(report),flush=True)


def client(args):
    args.output.mkdir(exist_ok=False)
    args.created_output = True
    protocol = dict(created_utc=utc(),modes=list(MODES),runs=len(MODES),dt=.1,max_steps=3000,
        original_gains=list(PID_GAINS),slot_order=list(SLOT_ORDER),pid_gains_encrypted=True,
        optimization='One packed ciphertext [e,I,d] times pre-encrypted packed gains [Kp,Ki,Kd]; final slot sum on cloud.',
        comparison='Reported CKKS-PID implementation; one full-course run.',
        context=dict(N=8192,modulus_bits=[60,40,40,60],scale=2**40,n_threads=1),
        initial_state=[-180.34,79.54,20,0],integral='20 errors times0.1 clipped[-20,20]',
        derivative='raw difference divided by0.1; zero first sample; no e/d input clipping for either PID',
        route='original full course, nearest-waypoint progress >=len-5 and updated endpoint distance<4m',
        actuation='asynchronous, one request in flight, apply available response at sampling boundary; hold previous if pending',
        warmup_requests_per_mode=20,timing='total request includes encryption, serialization, validated envelope, communication, cloud processing, return decode/decrypt; excludes setup, worker queue, saturation, and actuation wait',
        cloud_timing='server_ms contains validation, ciphertext decode, arithmetic, output serialization; evaluate_ms isolates arithmetic',
        privacy='encrypted-mode per-request exact field audit at client and server; no plaintext error or input; PID gains remain encrypted operands; published benchmark gain values are not confidential',
        load='Record host load around the formal run; timings depend on the execution environment.',
        hashes=hashes(),local_machine=machine())
    old = json.loads(Path('pid_twohost_fullcourse/protocol.json').read_text())
    assert protocol['hashes']['racetrack_waypoints.txt']==old['route_sha256']
    write_json(args.output/'protocol.json',protocol)
    coeff = np.load('revision_results/piecewise_coefficients.npy')
    assert coeff.shape==(384,10)
    context = ts.context(ts.SCHEME_TYPE.CKKS,8192,coeff_mod_bit_sizes=[60,40,40,60],n_threads=1)
    context.global_scale = 2**40
    context.generate_galois_keys()
    summary = []
    with socket.create_connection((args.host,args.port),timeout=30) as sock:
        sock.settimeout(120)
        sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
        send(sock,dict(hashes=hashes(),context=context.serialize(save_secret_key=False),
             scalar_gains=[ts.ckks_vector(context,[v]).serialize() for v in PID_GAINS],
             packed_gains=pack_gains(context).serialize()))
        ack,_ = recv(sock)
        assert ack['secret_key'] is False and ack['hashes']==hashes()
        write_json(args.output/'deployment_audit.json',dict(local_hashes=hashes(),remote=ack,matched=True))
        session = Session(sock,context)
        preflight(session,coeff,args)
        route = load_route()
        for mode in MODES:
            folder=args.output/mode
            folder.mkdir()
            send(sock,dict(command='snapshot'))
            before,_=recv(sock)
            write_json(folder/'load_before.json',dict(remote=before,local=machine(),utc=utc()))
            session.rows=[]
            warm_started=time.perf_counter()
            for k in range(20):
                remain=warm_started+k*.1-time.perf_counter()
                if remain>0:
                    time.sleep(remain)
                session.request(mode,k,[.2,-.1,.3],'warmup')
            save_csv(folder/'warmup.csv',session.rows)
            session.rows=[]
            print('RUN_START',mode,utc(),flush=True)
            trace,completed=full_course(session,mode,route,folder,args)
            send(sock,dict(command='snapshot'))
            after,_=recv(sock)
            write_json(folder/'load_after.json',dict(remote=after,local=machine(),utc=utc()))
            e=np.array([r['error'] for r in trace]);lat=np.array([r['total_ms'] for r in session.rows])
            age=[r['step']-r['applied_seq'] for r in trace if r['applied_seq']>=0]
            row=dict(controller=mode,completed=completed,steps=len(trace),requests=len(lat),
                 RMSE=float(np.sqrt(np.mean(e*e))),MAE=float(np.mean(abs(e))),Max_Error=float(max(abs(e))),Overshoot=float(max(0,-min(e))),
                 mean_ms=float(lat.mean()),p95_ms=float(np.percentile(lat,95)),p99_ms=float(np.percentile(lat,99)),max_ms=float(lat.max()),over100=int(sum(lat>100)),
                 sample_to_response_over100=sum(r['sample_to_response_ms']>100 for r in session.rows),
                 mean_server_ms=float(np.mean([r['server_ms'] for r in session.rows])),
                 mean_server_evaluate_ms=float(np.mean([r['server_evaluate_ms'] for r in session.rows])),
                 mean_encrypt_ms=float(np.mean([r['encrypt_ms'] for r in session.rows])),
                 mean_serialize_envelope_ms=float(np.mean([r['serialize_envelope_ms'] for r in session.rows])),
                 mean_up_bytes=float(np.mean([r['up_bytes'] for r in session.rows])),mean_down_bytes=float(np.mean([r['down_bytes'] for r in session.rows])),
                 applied_age_samples=','.join(map(str,sorted(set(age)))),skipped_submissions=sum(not r['request_sent'] for r in trace))
            summary.append(row)
            assert len(session.rows)==sum(r['request_sent'] for r in trace)
            assert after['accepted'][mode]-before['accepted'][mode]==20+len(session.rows)
            write_json(folder/'audit.json',dict(passed=True,formal_requests=len(session.rows),warmup=20,dual_field_validation=True,server_request_count_matched=True))
            save_csv(args.output/'summary.csv',summary)
            print('RUN_DONE',json.dumps(row),flush=True)
        send(sock,None)
    write_json(args.output/'complete.json',dict(runs=len(MODES),all_completed=all(r['completed'] for r in summary),finished_utc=utc()))
    write_json(args.output/'status.json',dict(state='complete',runs=len(MODES),updated_utc=utc()))
    print('COMPLETE',utc(),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--server',action='store_true')
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=47862)
    parser.add_argument('--output',type=Path,default=Path('packed_pid_twohost'))
    args=parser.parse_args()
    try:
        server(args) if args.server else client(args)
    except Exception:
        detail=traceback.format_exc()
        if not args.server and getattr(args,'created_output',False):
            write_json(args.output/'failure.json',dict(utc=utc(),error=detail))
        print(detail,file=sys.stderr,flush=True)
        sys.exit(1)
