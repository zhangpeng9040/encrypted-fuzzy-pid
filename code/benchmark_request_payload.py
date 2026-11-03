"""Strict request envelopes for the trusted-LAN controller benchmark.

This checks permitted plaintext fields. It is not message authentication and does
not make pickle safe for untrusted peers.
"""
import pickle
import struct

SCHEMAS = {
    'chen': frozenset(('mode', 'region', 'error')),
    'fuzzy': frozenset(('mode', 'region', 'z')),
    'poly': frozenset(('mode', 'region', 'z')),
    'ckks_poly': frozenset(('mode', 'region', 'blob')),
}

def validate_request(request):
    if not isinstance(request, dict):
        raise ValueError('request must be a dictionary')
    mode = request.get('mode')
    if mode not in SCHEMAS or set(request) != SCHEMAS[mode]:
        raise ValueError('unexpected request fields')
    if not isinstance(request['region'], int):
        raise ValueError('region must be an integer')
    if mode == 'ckks_poly':
        if not isinstance(request['blob'], bytes) or not request['blob']:
            raise ValueError('encrypted request requires serialized ciphertext bytes')
    elif mode == 'chen':
        if not isinstance(request['error'], float):
            raise ValueError('Chen requires a plaintext scalar error')
    elif not (isinstance(request['z'], list) and len(request['z']) == 3
              and all(isinstance(v, float) for v in request['z'])):
        raise ValueError('plaintext polynomial/fuzzy request requires three floats')
    return mode

def build_request(mode, region, z, error, encrypt):
    request = dict(mode=mode, region=int(region))
    if mode == 'chen':
        request['error'] = float(error)
    elif mode in ('fuzzy', 'poly'):
        request['z'] = [float(v) for v in z]
    elif mode == 'ckks_poly':
        request['blob'] = encrypt(z)
    else:
        raise ValueError('unknown controller mode')
    validate_request(request)
    return request

def send_checked_request(sock, request):
    # Audit the exact serialized payload that is subsequently sent, not a copy
    # constructed independently for testing or logging.
    body = pickle.dumps(request, protocol=5)
    decoded = pickle.loads(body)
    validate_request(decoded)
    sock.sendall(struct.pack('!I', len(body)) + body)
    return len(body) + 4, ','.join(sorted(decoded))
