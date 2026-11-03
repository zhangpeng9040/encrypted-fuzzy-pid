#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模糊PID控制器（纵向）- 积分器改为有限长（窗口长度20）
- 明文：Mamdani 推理动态生成 Kp, Ki, Kd → PID 控制
- 密文：分区三维二次多项式逼近油门曲面（输入 e、de、积分 ie_sum，深度1）
横向控制：Stanley（明文）

【本版修改】
  1. update_desired_speed 去掉 "desired>9 则减9" 阶跃。
  2. 积分改为有限长：使用 deque(maxlen=20) 存储误差，积分值 = sum(buffer) * DT。
  3. 积分抗饱和到 ie_range；在 _pid_throttle 中把 e/de/ie clamp 到训练域。
  4. 加密版加密前同样 clamp e/de/ie。
"""

import numpy as np
from lateral_control import steering_radians
from collections import deque
import tenseal as ts
from itertools import product

DT = 0.1


class FuzzyLongitudinalController:
    def __init__(self, use_encrypted=True):
        self.use_encrypted = use_encrypted

        # ========== 明文模糊参数 ==========
        self.e_mf_params = {
            'NB': (-10, -10, -6, -3), 'NM': (-6, -3, -1.5, -0.5),
            'NS': (-1.5, -0.5, -0.2, 0), 'ZE': (-0.2, 0, 0, 0.2),
            'PS': (0, 0.2, 0.5, 1.5), 'PM': (0.5, 1.5, 3, 6), 'PB': (3, 6, 10, 10)}
        self.de_mf_params = {
            'NB': (-2, -2, -1.2, -0.6), 'NM': (-1.2, -0.6, -0.3, -0.1),
            'NS': (-0.3, -0.1, -0.05, 0), 'ZE': (-0.05, 0, 0, 0.05),
            'PS': (0, 0.05, 0.1, 0.3), 'PM': (0.1, 0.3, 0.6, 1.2), 'PB': (0.6, 1.2, 2, 2)}
        self.u_mf_params = {
            'NB': (-1, -1, -0.8, -0.5), 'NM': (-0.8, -0.5, -0.3, -0.1),
            'NS': (-0.3, -0.1, -0.05, 0), 'ZE': (-0.05, 0, 0, 0.05),
            'PS': (0, 0.05, 0.1, 0.3), 'PM': (0.1, 0.3, 0.5, 0.8), 'PB': (0.5, 0.8, 1, 1)}
        self.rule_base = {
            ('NB','NB'):'NB',('NB','NM'):'NB',('NB','NS'):'NM',('NB','ZE'):'NM',
            ('NB','PS'):'NS',('NB','PM'):'NS',('NB','PB'):'ZE',
            ('NM','NB'):'NB',('NM','NM'):'NM',('NM','NS'):'NM',('NM','ZE'):'NS',
            ('NM','PS'):'NS',('NM','PM'):'ZE',('NM','PB'):'PS',
            ('NS','NB'):'NM',('NS','NM'):'NM',('NS','NS'):'NS',('NS','ZE'):'NS',
            ('NS','PS'):'ZE',('NS','PM'):'PS',('NS','PB'):'PM',
            ('ZE','NB'):'NM',('ZE','NM'):'NS',('ZE','NS'):'NS',('ZE','ZE'):'ZE',
            ('ZE','PS'):'PS',('ZE','PM'):'PS',('ZE','PB'):'PM',
            ('PS','NB'):'NS',('PS','NM'):'NS',('PS','NS'):'ZE',('PS','ZE'):'PS',
            ('PS','PS'):'PS',('PS','PM'):'PM',('PS','PB'):'PB',
            ('PM','NB'):'NS',('PM','NM'):'ZE',('PM','NS'):'PS',('PM','ZE'):'PS',
            ('PM','PS'):'PM',('PM','PM'):'PM',('PM','PB'):'PB',
            ('PB','NB'):'ZE',('PB','NM'):'PS',('PB','NS'):'PS',('PB','ZE'):'PM',
            ('PB','PS'):'PM',('PB','PM'):'PB',('PB','PB'):'PB'}
        self.e_buffer = deque(maxlen=20)
        self.u_centroids = {l: (a+b+c+d)/4.0 for l, (a,b,c,d) in self.u_mf_params.items()}

        # PID 基值增益（已适配本平台：v += u*a*dt, a=3 → 回路增益 a*dt*Kp ≈ 0.3，稳定）
        self.Kp_base = 1.0
        self.Ki_base = 0.3
        self.Kd_base = 0.001

        # ========== 积分器：有限长滑动窗口（长度20） ==========
        self.ie_buffer = deque(maxlen=20)   # 存储误差 e，积分值 = sum(ie_buffer) * DT

        # ========== CKKS 上下文（深度1） ==========
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60])
        self.context.generate_galois_keys()
        self.context.global_scale = 2 ** 40

        # ========== 三维分区 ==========
        self.e_bins = [-10, -6, -3, 0, 3, 6, 10]
        self.de_bins = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
        self.ie_bins = [-10.0, -5.0, 0.0, 5.0, 10.0]
        self.num_regions = (len(self.e_bins)+1) * (len(self.de_bins)+1) * (len(self.ie_bins)+1)
        self.e_range = (-15, 15)
        self.de_range = (-3, 3)      # 与 de 隶属函数饱和域(±2)匹配
        self.ie_range = (-20, 20)    # 窗口积分和范围（最大约 ±20）

        self.local_coeffs = self._fit_all_regions()
        print(f"✅ 分区多项式已就绪（{self.num_regions} 个区域）")

    def _get_ie_sum(self):
        """计算当前滑动窗口积分和 = sum(errors) * DT"""
        return sum(self.ie_buffer) * DT

    # ==================== 明文基础模糊推理 ====================
    def _trapezoid_mf(self, x, params):
        a, b, c, d = params
        if a == b and x <= b: return 1.0
        if c == d and x >= c: return 1.0
        if x <= a or x >= d: return 0.0
        if a < x < b: return (x - a) / (b - a) if (b - a) > 1e-6 else 0.0
        if b <= x <= c: return 1.0
        return (d - x) / (d - c) if (d - c) > 1e-6 else 0.0

    def _fuzzify(self, value, mf_params):
        value = float(np.clip(value, min(p[0] for p in mf_params.values()), max(p[3] for p in mf_params.values())))
        return {l: self._trapezoid_mf(value, p) for l, p in mf_params.items()}

    def _inference(self, e_mem, de_mem):
        act = {l: 0.0 for l in self.u_mf_params}
        for e_l, e_deg in e_mem.items():
            if e_deg < 1e-6: continue
            for de_l, de_deg in de_mem.items():
                if de_deg < 1e-6: continue
                rule_act = min(e_deg, de_deg)
                out_l = self.rule_base[(e_l, de_l)]
                act[out_l] = max(act[out_l], rule_act)
        return act

    def _defuzzify(self, act):
        num = den = 0.0
        for label, strength in act.items():
            if strength < 1e-6: continue
            a, b, c, d = self.u_mf_params[label]
            num += strength * ((a+b+c+d)/4.0)
            den += strength
        return num / den if den > 1e-6 else 0.0

    def _plain_throttle(self, e, de):
        e_mem = self._fuzzify(e, self.e_mf_params)
        de_mem = self._fuzzify(de, self.de_mf_params)
        return np.clip(self._defuzzify(self._inference(e_mem, de_mem)), -1.0, 1.0)

    # ==================== 增益调度 ====================
    def _plain_gains(self, e, de):
        u_raw = self._plain_throttle(e, de)
        kp = self.Kp_base * (1.0 + 0.5 * u_raw)
        ki = self.Ki_base * (1.0 - 0.3 * u_raw)
        kd = self.Kd_base * (1.0 + 0.2 * abs(de))
        kp = np.clip(kp, 0.5*self.Kp_base, 1.5*self.Kp_base)
        ki = np.clip(ki, 0.5*self.Ki_base, 1.5*self.Ki_base)
        kd = np.clip(kd, 0.5*self.Kd_base, 1.5*self.Kd_base)
        return kp, ki, kd

    # ==================== 模糊 PID 油门 ====================
    def _pid_throttle(self, e, de, ie_sum=None):
        """
        计算油门输出
        :param e: 速度误差
        :param de: 误差微分
        :param ie_sum: 积分和（若为 None 则使用当前滑动窗口积分）
        """
        if ie_sum is None:
            ie_sum = self._get_ie_sum()
        # clamp 到训练域，保证明文与加密(多项式)输入域一致
        e_c = float(np.clip(e, self.e_range[0], self.e_range[1]))
        de_c = float(np.clip(de, self.de_range[0], self.de_range[1]))
        ie_c = float(np.clip(ie_sum, self.ie_range[0], self.ie_range[1]))
        kp, ki, kd = self._plain_gains(e_c, de_c)
        u = kp * e_c + ki * ie_c + kd * de_c
        return np.clip(u, -1.0, 1.0)

    # ==================== 分区多项式训练 ====================
    def _fit_all_regions(self):
        coeffs_list = []
        e_edges = [-np.inf] + self.e_bins + [np.inf]
        de_edges = [-np.inf] + self.de_bins + [np.inf]
        ie_edges = [-np.inf] + self.ie_bins + [np.inf]
        for i in range(len(e_edges) - 1):
            e_min = max(self.e_range[0], e_edges[i]); e_max = min(self.e_range[1], e_edges[i+1])
            if e_max <= e_min: e_max = e_min + 0.1
            for j in range(len(de_edges) - 1):
                de_min = max(self.de_range[0], de_edges[j]); de_max = min(self.de_range[1], de_edges[j+1])
                if de_max <= de_min: de_max = de_min + 0.1
                for k in range(len(ie_edges) - 1):
                    ie_min = max(self.ie_range[0], ie_edges[k]); ie_max = min(self.ie_range[1], ie_edges[k+1])
                    if ie_max <= ie_min: ie_max = ie_min + 0.1
                    e_vals = np.linspace(e_min, e_max, 10)
                    de_vals = np.linspace(de_min, de_max, 8)
                    ie_vals = np.linspace(ie_min, ie_max, 7)
                    X, y = [], []
                    for e, de, ie_sum in product(e_vals, de_vals, ie_vals):
                        X.append([1.0, e, de, ie_sum, e**2, e*de, e*ie_sum, de**2, de*ie_sum, ie_sum**2])
                        y.append(self._pid_throttle(e, de, ie_sum=ie_sum))
                    coeffs, _, _, _ = np.linalg.lstsq(np.array(X), np.array(y), rcond=None)
                    coeffs_list.append(coeffs)
        return coeffs_list

    def _region_index(self, e, de, ie_sum=0.0):
        e_idx = 0
        for i, b in enumerate(self.e_bins):
            if e >= b: e_idx = i + 1
        de_idx = 0
        for j, b in enumerate(self.de_bins):
            if de >= b: de_idx = j + 1
        ie_idx = 0
        for k, b in enumerate(self.ie_bins):
            if ie_sum >= b: ie_idx = k + 1
        de_cnt = len(self.de_bins) + 1
        ie_cnt = len(self.ie_bins) + 1
        return (e_idx * de_cnt + de_idx) * ie_cnt + ie_idx

    def _polyval_encrypted(self, enc_e, enc_de, enc_ie, coeffs):
        e2 = enc_e * enc_e
        de2 = enc_de * enc_de
        ie2 = enc_ie * enc_ie
        ed = enc_e * enc_de
        ei = enc_e * enc_ie
        dei = enc_de * enc_ie
        return (e2 * coeffs[4] + ed * coeffs[5] + ei * coeffs[6] +
                de2 * coeffs[7] + dei * coeffs[8] + ie2 * coeffs[9] +
                enc_e * coeffs[1] + enc_de * coeffs[2] + enc_ie * coeffs[3] + coeffs[0])

    @staticmethod
    def polynomial_features(e, de, ie):
        return np.array([1.0, e, de, ie, e*e, e*de, e*ie, de*de, de*ie, ie*ie])

    def evaluate_plain_polynomial(self, e, de, ie):
        """Same coefficients, region selection and clipping as encrypted evaluation."""
        e, de, ie = [float(np.clip(v, *r)) for v, r in
                     zip((e, de, ie), (self.e_range, self.de_range, self.ie_range))]
        idx = self._region_index(e, de, ie)
        return float(np.clip(self.polynomial_features(e, de, ie) @ self.local_coeffs[idx], -1, 1))

    def compute_throttle_polynomial(self, speed_error):
        self.e_buffer.append(speed_error)
        self.ie_buffer.append(speed_error)
        de = (self.e_buffer[-1] - self.e_buffer[-2]) / DT if len(self.e_buffer) >= 2 else 0.0
        return self.evaluate_plain_polynomial(speed_error, de, self._get_ie_sum())

    # ==================== 公开接口 ====================
    def compute_throttle(self, speed_error):
        """明文模糊 PID 控制（有限长积分）"""
        self.e_buffer.append(speed_error)
        # 更新积分滑动窗口
        self.ie_buffer.append(speed_error)
        ie_sum = self._get_ie_sum()
        ie_sum = float(np.clip(ie_sum, self.ie_range[0], self.ie_range[1]))
        if len(self.e_buffer) >= 2:
            de = (self.e_buffer[-1] - self.e_buffer[-2]) / DT
        else:
            de = 0.0
        return self._pid_throttle(speed_error, de, ie_sum=ie_sum)

    def compute_throttle_encrypted(self, speed_error, delta_error=None):
        """加密模糊控制器：三维多项式逼近，输入 e、de、ie_sum（车端明文 clamp 后加密）"""
        if delta_error is None:
            self.e_buffer.append(speed_error)
            delta_error = (self.e_buffer[-1] - self.e_buffer[-2]) / DT if len(self.e_buffer) >= 2 else 0.0

        # 更新积分滑动窗口
        self.ie_buffer.append(speed_error)
        ie_sum = self._get_ie_sum()
        ie_sum = float(np.clip(ie_sum, self.ie_range[0], self.ie_range[1]))

        # 车端明文 clamp 到训练域
        e_c = float(np.clip(speed_error, self.e_range[0], self.e_range[1]))
        de_c = float(np.clip(delta_error, self.de_range[0], self.de_range[1]))
        ie_c = float(np.clip(ie_sum, self.ie_range[0], self.ie_range[1]))

        idx = self._region_index(e_c, de_c, ie_c)
        coeffs = self.local_coeffs[idx]
        enc_e = ts.ckks_vector(self.context, [e_c])
        enc_de = ts.ckks_vector(self.context, [de_c])
        enc_ie = ts.ckks_vector(self.context, [ie_c])
        enc_throttle = self._polyval_encrypted(enc_e, enc_de, enc_ie, coeffs)
        dec = enc_throttle.decrypt()
        throttle_poly = dec[0] if len(dec) > 0 else 0.0
        return np.clip(throttle_poly, -1.0, 1.0)


# ==================== 2D 控制器封装 ====================
class FuzzyController2D:
    def __init__(self, waypoints, encrypted=False, polynomial=False):
        self._current_x = 0.0
        self._current_y = 0.0
        self._current_yaw = 0.0
        self._current_speed = 0.0
        self._desired_speed = 0.0
        self.throttle = 0.0
        self.brake = 0.0
        self.steer = 0.0
        self._waypoints = waypoints
        self._conv_rad_to_steer = 180.0 / 70.0 / np.pi
        self._pi = np.pi
        self._2pi = 2.0 * np.pi

        self.fuzzy_longitudinal = FuzzyLongitudinalController(use_encrypted=encrypted)
        self.encrypted = encrypted
        self.polynomial = polynomial
        print(f"✅ FuzzyController2D 初始化完成 (加密模式 = {encrypted})")

    def update_values(self, x, y, yaw, speed):
        self._current_x = x
        self._current_y = y
        self._current_yaw = yaw
        self._current_speed = speed

    def update_desired_speed(self):
        """直接使用航点速度（已去掉 -9 阶跃）。"""
        diff = self._waypoints[:, :2] - np.array([self._current_x, self._current_y])
        dists = np.sqrt((diff ** 2).sum(axis=1))
        min_idx = int(np.argmin(dists))
        if min_idx < len(self._waypoints) - 1:
            self._desired_speed = self._waypoints[min_idx, 2]
        else:
            self._desired_speed = self._waypoints[-1, 2]

    def update_waypoints(self, new_waypoints):
        self._waypoints = np.array(new_waypoints)

    def update_controls(self):
        x, y, yaw, v = self._current_x, self._current_y, self._current_yaw, self._current_speed
        self.update_desired_speed()
        speed_error = self._desired_speed - v

        if self.encrypted:
            self.throttle = self.fuzzy_longitudinal.compute_throttle_encrypted(speed_error)
        elif self.polynomial:
            self.throttle = self.fuzzy_longitudinal.compute_throttle_polynomial(speed_error)
        else:
            self.throttle = self.fuzzy_longitudinal.compute_throttle(speed_error)

        # 横向 Stanley
        k_e, k_v = 0.3, 20
        yaw_path = np.arctan2(self._waypoints[-1, 1] - self._waypoints[0, 1],
                              self._waypoints[-1, 0] - self._waypoints[0, 0])
        yaw_diff = yaw_path - yaw
        if yaw_diff > np.pi: yaw_diff -= 2 * np.pi
        if yaw_diff < -np.pi: yaw_diff += 2 * np.pi

        crosstrack = np.min(np.sum((np.array([x, y]) - self._waypoints[:, :2]) ** 2, axis=1))
        yaw_ct = np.arctan2(y - self._waypoints[0, 1], x - self._waypoints[0, 0])
        path2ct = yaw_path - yaw_ct
        if path2ct > np.pi: path2ct -= 2 * np.pi
        if path2ct < -np.pi: path2ct += 2 * np.pi
        crosstrack = abs(crosstrack) if path2ct > 0 else -abs(crosstrack)

        steer_exp = yaw_diff + np.arctan(k_e * crosstrack / (k_v + v))
        if steer_exp > np.pi: steer_exp -= 2 * np.pi
        if steer_exp < -np.pi: steer_exp += 2 * np.pi
        steer_exp = np.clip(steer_exp, -1.22, 1.22)
        self.steer = steering_radians(self._current_x, self._current_y, self._current_yaw, self._current_speed, self._waypoints)


# ==================== 简单测试 ====================
if __name__ == "__main__":
    waypts = np.array([[0, 0, 10], [10, 0, 10], [20, 5, 8], [30, 10, 6]])
    ctrl_plain = FuzzyController2D(waypts, encrypted=False)
    ctrl_plain.update_values(5, 0.5, 0.1, 8)
    ctrl_plain.update_controls()
    print(f"明文 throttle = {ctrl_plain.throttle:.4f}")
    ctrl_enc = FuzzyController2D(waypts, encrypted=True)
    ctrl_enc.update_values(5, 0.5, 0.1, 8)
    ctrl_enc.update_controls()
    print(f"密文 throttle = {ctrl_enc.throttle:.4f}")
    print("✅ 测试完成")