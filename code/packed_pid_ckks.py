"""Packed CKKS PID with encrypted gains and cloud-side final summation.

Inputs to this module are (error, derivative, integral); encrypted slots are
(error, integral, derivative), paired with (Kp, Ki, Kd).
"""
import numpy as np
import tenseal as ts

PID_GAINS = (1.0, 0.3, 0.001)
SLOT_ORDER = ('e', 'I', 'd')


def pack_inputs(context, z):
    e, d, integral = map(float, z)
    if not np.isfinite([e, d, integral]).all():
        raise ValueError('PID input must be finite')
    return ts.ckks_vector(context, [e, integral, d])


def pack_gains(context, gains=PID_GAINS):
    if len(gains) != 3 or not np.isfinite(gains).all():
        raise ValueError('PID requires three finite gains')
    return ts.ckks_vector(context, list(map(float, gains)))


def evaluate_packed(inputs, encrypted_gains):
    if inputs.size() != 3 or encrypted_gains.size() != 3:
        raise ValueError('PID ciphertexts must each contain three logical slots')
    return (inputs * encrypted_gains).sum()


def validate_request(request):
    if not isinstance(request, dict):
        raise ValueError('request must be a dictionary')
    mode = request.get('mode')
    schemas = {'pid_scalar': {'mode', 'blobs'}, 'pid_packed': {'mode', 'blob'},
               'ckks_poly': {'mode', 'region', 'blob'}}
    if mode not in schemas or set(request) != schemas[mode]:
        raise ValueError('unexpected fields in encrypted control request')
    if mode == 'pid_scalar':
        if not (type(request['blobs']) is list and len(request['blobs']) == 3
                and all(type(b) is bytes and b for b in request['blobs'])):
            raise ValueError('scalar PID requires three serialized ciphertexts')
    elif type(request['blob']) is not bytes or not request['blob']:
        raise ValueError('packed request requires one serialized ciphertext')
    if mode == 'ckks_poly' and not (type(request['region']) is int and 0 <= request['region'] < 384):
        raise ValueError('invalid public polynomial cell index')
    return mode
