#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TASE2026 模糊PID控制器（明文 + 加密版）
- 明文：复现 IEEE TASE 2026 论文中的模糊PID（5级模糊集，三角形隶属函数，重心法去模糊化）
- 加密：三维分区二次多项式近似（CKKS），输入 e, de, I
横向控制：Stanley（与原有框架一致）
- 【修改】积分器改为滑动窗口（长度20），与 controller2d.py 行为一致
"""

import numpy as np
from lateral_control import steering_radians
import tenseal as ts
from itertools import product
from collections import deque

DT = 0.1  # 仿真步长

# ==================== 明文版 TASE2026 模糊PID ====================
class Tase2026FuzzyController:
    """
    完整 2D 控制器：纵向为 TASE2026 模糊 PID，横向为 Stanley
    """
    def __init__(self, waypoints):
        self._current_x = 0.0
        self._current_y = 0.0
        self._current_yaw = 0.0
        self._current_speed = 0.0
        self._desired_speed = 0.0
        self.throttle = 0.0
        self.steer = 0.0
        self._waypoints = waypoints
        self._conv_rad_to_steer = 180.0 / 70.0 / np.pi
        self._pi = np.pi
        self._2pi = 2.0 * np.pi

        # 纵向模糊 PID 控制器实例（滑动窗口积分）
        self.longitudinal = _Tase2026Longitudinal()
        print("✅ TASE2026 明文模糊PID控制器初始化完成（积分窗口长度20）")

    # ---------- 公共接口 ----------
    def update_values(self, x, y, yaw, speed):
        self._current_x = x
        self._current_y = y
        self._current_yaw = yaw
        self._current_speed = speed

    def update_desired_speed(self):
        diff = self._waypoints[:, :2] - np.array([self._current_x, self._current_y])
        dists = np.sqrt((diff ** 2).sum(axis=1))
        min_idx = int(np.argmin(dists))
        if min_idx < len(self._waypoints) - 1:
            desired = self._waypoints[min_idx, 2]
        else:
            desired = self._waypoints[-1, 2]
        self._desired_speed = desired  # 直接使用航点速度（已去掉减9阶跃）

    def update_waypoints(self, new_waypoints):
        self._waypoints = np.array(new_waypoints)

    def update_controls(self):
        self.update_desired_speed()
        speed_error = self._desired_speed - self._current_speed
        self.throttle = self.longitudinal.compute_throttle(speed_error)
        self._lateral_control()

    def _lateral_control(self):
        """Stanley 横向控制（与原有框架完全一致）"""
        x = self._current_x
        y = self._current_y
        yaw = self._current_yaw
        v = self._current_speed
        waypoints = self._waypoints
        k_e, k_v = 0.3, 20

        yaw_path = np.arctan2(waypoints[-1][1] - waypoints[0][1],
                              waypoints[-1][0] - waypoints[0][0])
        yaw_diff = yaw_path - yaw
        if yaw_diff > np.pi:
            yaw_diff -= 2 * np.pi
        if yaw_diff < -np.pi:
            yaw_diff += 2 * np.pi

        current_xy = np.array([x, y])
        crosstrack_error = np.min(np.sum((current_xy - waypoints[:, :2]) ** 2, axis=1))
        yaw_cross_track = np.arctan2(y - waypoints[0][1], x - waypoints[0][0])
        yaw_path2ct = yaw_path - yaw_cross_track
        if yaw_path2ct > np.pi:
            yaw_path2ct -= 2 * np.pi
        if yaw_path2ct < -np.pi:
            yaw_path2ct += 2 * np.pi
        if yaw_path2ct > 0:
            crosstrack_error = abs(crosstrack_error)
        else:
            crosstrack_error = -abs(crosstrack_error)

        yaw_diff_crosstrack = np.arctan(k_e * crosstrack_error / (k_v + v))
        steer_expect = yaw_diff + yaw_diff_crosstrack
        if steer_expect > np.pi:
            steer_expect -= 2 * np.pi
        if steer_expect < -np.pi:
            steer_expect += 2 * np.pi
        steer_expect = np.clip(steer_expect, -1.22, 1.22)

        self.steer = steering_radians(self._current_x, self._current_y, self._current_yaw, self._current_speed, self._waypoints)


# ---------- 内部纵向控制器（明文模糊PID，滑动窗口积分） ----------
class _Tase2026Longitudinal:
    """
    TASE2026 纵向模糊 PID（5级模糊集，三角形隶属函数，重心法去模糊化）
    输入：速度误差 e (m/s)
    输出：油门/刹车指令 [-1,1]
    积分器：滑动窗口长度20（即最近2秒的误差累加）
    """
    def __init__(self):
        # 输入论域与量化因子（适配本车辆模型：速度误差范围 ±16 m/s，变化率 ±3 m/s²）
        self.e_range = (-16, 16)
        self.ec_range = (-3, 3)
        self.e_qf = 4.0 / 16.0   # 量化到 [-4,4]
        self.ec_qf = 4.0 / 3.0   # 量化到 [-4,4]

        # 模糊集标签（5级）
        self.labels = ['NB', 'NS', 'ZO', 'PS', 'PB']
        # 三角形隶属函数中心（量化后的论域上）
        self.centers = {'NB': -2.0, 'NS': -1.0, 'ZO': 0.0, 'PS': 1.0, 'PB': 2.0}
        self.half_width = 1.0   # 相邻中心间距的一半

        # 输出修正量论域中心（经过适配本平台）
        self.delta_kp_centers = {'NB': -4.0, 'NS': -2.0, 'ZO': 0.0, 'PS': 2.0, 'PB': 4.0}
        self.delta_ki_centers = {'NB': -0.04, 'NS': -0.02, 'ZO': 0.0, 'PS': 0.02, 'PB': 0.04}
        self.delta_kd_centers = {'NB': -1.5, 'NS': -0.75, 'ZO': 0.0, 'PS': 0.75, 'PB': 1.5}
        self.delta_kp_list = [self.delta_kp_centers[l] for l in self.labels]
        self.delta_ki_list = [self.delta_ki_centers[l] for l in self.labels]
        self.delta_kd_list = [self.delta_kd_centers[l] for l in self.labels]

        # PID 初始基值（与您的模糊PID基值一致，保证对比公平）
        self.Kp0 = 1.0
        self.Ki0 = 0.3
        self.Kd0 = 0.001

        # 滑动窗口积分器（长度20）
        self.ie_buffer = deque(maxlen=20)   # 存储误差 e
        self.prev_error = 0.0
        self.dt = DT

        # 模糊规则表（论文 Table III 用于 ΔKp 和 ΔKd，Table IV 用于 ΔKi）
        # 5x5 矩阵，行列对应 e 和 ec 的模糊集索引（顺序 NB, NS, ZO, PS, PB）
        rule_kp_kd = [
            ['NB', 'NB', 'NB', 'NS', 'NS'],
            ['NB', 'NB', 'NS', 'NS', 'NS'],
            ['NS', 'NS', 'ZO', 'PS', 'PS'],
            ['PS', 'PS', 'PS', 'PB', 'PB'],
            ['PS', 'PS', 'PB', 'PB', 'PB']
        ]
        rule_ki = [
            ['ZO', 'ZO', 'ZO', 'ZO', 'ZO'],
            ['ZO', 'ZO', 'ZO', 'ZO', 'ZO'],
            ['PS', 'PS', 'PB', 'PB', 'PB'],
            ['ZO', 'ZO', 'ZO', 'ZO', 'ZO'],
            ['ZO', 'ZO', 'ZO', 'ZO', 'ZO']
        ]
        # 转换为索引矩阵
        self.rule_kp_idx = [[self.labels.index(cell) for cell in row] for row in rule_kp_kd]
        self.rule_kd_idx = self.rule_kp_idx
        self.rule_ki_idx = [[self.labels.index(cell) for cell in row] for row in rule_ki]

    def _get_integral(self):
        """计算当前滑动窗口积分和 = sum(errors) * DT，并限幅到 [-20,20]"""
        ie = sum(self.ie_buffer) * self.dt
        return np.clip(ie, -20.0, 20.0)

    # ---------- 模糊推理基础函数 ----------
    def _triangle_mf(self, x, center, half_width):
        if x <= center - half_width or x >= center + half_width:
            return 0.0
        if x <= center:
            return (x - (center - half_width)) / half_width
        else:
            return ((center + half_width) - x) / half_width

    def _fuzzify(self, x, centers, half_width):
        return {l: self._triangle_mf(x, c, half_width) for l, c in centers.items()}

    def _inference(self, mu_e, mu_ec, rule_idx):
        activation = {i: 0.0 for i in range(len(self.labels))}
        for i, e_label in enumerate(self.labels):
            me = mu_e[e_label]
            if me < 1e-6:
                continue
            for j, ec_label in enumerate(self.labels):
                mec = mu_ec[ec_label]
                if mec < 1e-6:
                    continue
                w = min(me, mec)
                out_idx = rule_idx[i][j]
                activation[out_idx] = max(activation[out_idx], w)
        return activation

    def _defuzzify(self, activation, centers_list):
        num = den = 0.0
        for idx, w in activation.items():
            if w < 1e-6:
                continue
            num += w * centers_list[idx]
            den += w
        return num / den if den > 1e-6 else 0.0

    # ---------- 公开接口 ----------
    def compute_throttle(self, speed_error):
        # 更新滑动窗口积分器
        self.ie_buffer.append(speed_error)
        integral = self._get_integral()

        # 误差变化率
        ec = (speed_error - self.prev_error) / self.dt
        self.prev_error = speed_error

        # 量化输入
        e_norm = np.clip(speed_error, self.e_range[0], self.e_range[1]) * self.e_qf
        ec_norm = np.clip(ec, self.ec_range[0], self.ec_range[1]) * self.ec_qf

        # 模糊化
        mu_e = self._fuzzify(e_norm, self.centers, self.half_width)
        mu_ec = self._fuzzify(ec_norm, self.centers, self.half_width)

        # 推理
        act_kp = self._inference(mu_e, mu_ec, self.rule_kp_idx)
        act_ki = self._inference(mu_e, mu_ec, self.rule_ki_idx)
        act_kd = self._inference(mu_e, mu_ec, self.rule_kd_idx)

        # 去模糊化
        delta_kp = self._defuzzify(act_kp, self.delta_kp_list)
        delta_ki = self._defuzzify(act_ki, self.delta_ki_list)
        delta_kd = self._defuzzify(act_kd, self.delta_kd_list)

        # PID 参数
        kp = self.Kp0 + delta_kp
        ki = self.Ki0 + delta_ki
        kd = self.Kd0 + delta_kd
        kp = np.clip(kp, 0.1, 5.0)
        ki = np.clip(ki, 0.0, 1.0)
        kd = np.clip(kd, 0.0, 5.0)

        # PID 控制律（使用滑动窗口积分值）
        u = kp * speed_error + ki * integral + kd * ec
        return np.clip(u, -1.0, 1.0)


# ==================== 训练用明文控制器（生成加密多项式训练数据） ====================
class Tase2026FuzzyTrain:
    """
    TASE2026 模糊PID（明文），用于生成加密多项式的训练目标（输入 e, de, ie）
    注：训练时积分值 ie 作为独立变量输入，与运行时滑动窗口计算方式数学一致，无需修改。
    """
    def __init__(self):
        self.e_range = (-16, 16)
        self.ec_range = (-3, 3)
        self.e_qf = 4.0 / 16.0
        self.ec_qf = 4.0 / 3.0

        self.labels = ['NB', 'NS', 'ZO', 'PS', 'PB']
        self.centers = {'NB': -2.0, 'NS': -1.0, 'ZO': 0.0, 'PS': 1.0, 'PB': 2.0}
        self.half_width = 1.0

        self.delta_kp_centers = {'NB': -4.0, 'NS': -2.0, 'ZO': 0.0, 'PS': 2.0, 'PB': 4.0}
        self.delta_ki_centers = {'NB': -0.04, 'NS': -0.02, 'ZO': 0.0, 'PS': 0.02, 'PB': 0.04}
        self.delta_kd_centers = {'NB': -1.5, 'NS': -0.75, 'ZO': 0.0, 'PS': 0.75, 'PB': 1.5}
        self.delta_kp_list = [self.delta_kp_centers[l] for l in self.labels]
        self.delta_ki_list = [self.delta_ki_centers[l] for l in self.labels]
        self.delta_kd_list = [self.delta_kd_centers[l] for l in self.labels]

        self.Kp0 = 1.0
        self.Ki0 = 0.3
        self.Kd0 = 0.001

        rule_kp_kd = [
            ['NB', 'NB', 'NB', 'NS', 'NS'],
            ['NB', 'NB', 'NS', 'NS', 'NS'],
            ['NS', 'NS', 'ZO', 'PS', 'PS'],
            ['PS', 'PS', 'PS', 'PB', 'PB'],
            ['PS', 'PS', 'PB', 'PB', 'PB']
        ]
        rule_ki = [
            ['ZO', 'ZO', 'ZO', 'ZO', 'ZO'],
            ['ZO', 'ZO', 'ZO', 'ZO', 'ZO'],
            ['PS', 'PS', 'PB', 'PB', 'PB'],
            ['ZO', 'ZO', 'ZO', 'ZO', 'ZO'],
            ['ZO', 'ZO', 'ZO', 'ZO', 'ZO']
        ]
        self.rule_kp_idx = [[self.labels.index(cell) for cell in row] for row in rule_kp_kd]
        self.rule_kd_idx = self.rule_kp_idx
        self.rule_ki_idx = [[self.labels.index(cell) for cell in row] for row in rule_ki]

    def _triangle_mf(self, x, center, half_width):
        if x <= center - half_width or x >= center + half_width:
            return 0.0
        if x <= center:
            return (x - (center - half_width)) / half_width
        else:
            return ((center + half_width) - x) / half_width

    def _fuzzify(self, x, centers, half_width):
        return {l: self._triangle_mf(x, c, half_width) for l, c in centers.items()}

    def _inference(self, mu_e, mu_ec, rule_idx):
        activation = {i: 0.0 for i in range(len(self.labels))}
        for i, e_label in enumerate(self.labels):
            me = mu_e[e_label]
            if me < 1e-6: continue
            for j, ec_label in enumerate(self.labels):
                mec = mu_ec[ec_label]
                if mec < 1e-6: continue
                w = min(me, mec)
                out_idx = rule_idx[i][j]
                activation[out_idx] = max(activation[out_idx], w)
        return activation

    def _defuzzify(self, activation, centers_list):
        num = den = 0.0
        for idx, w in activation.items():
            if w < 1e-6: continue
            num += w * centers_list[idx]
            den += w
        return num / den if den > 1e-6 else 0.0

    def compute_throttle_with_integral(self, speed_error, delta_error, integral):
        """
        给定 (e, de, ie) 的模糊PID输出油门（用于训练）
        integral 即积分和 I = sum(e)*dt，与运行时滑动窗口积分含义一致
        """
        e_c = float(np.clip(speed_error, self.e_range[0], self.e_range[1]))
        ec_c = float(np.clip(delta_error, self.ec_range[0], self.ec_range[1]))
        ie_c = float(np.clip(integral, -20.0, 20.0))

        e_norm = e_c * self.e_qf
        ec_norm = ec_c * self.ec_qf

        mu_e = self._fuzzify(e_norm, self.centers, self.half_width)
        mu_ec = self._fuzzify(ec_norm, self.centers, self.half_width)

        act_kp = self._inference(mu_e, mu_ec, self.rule_kp_idx)
        act_ki = self._inference(mu_e, mu_ec, self.rule_ki_idx)
        act_kd = self._inference(mu_e, mu_ec, self.rule_kd_idx)

        delta_kp = self._defuzzify(act_kp, self.delta_kp_list)
        delta_ki = self._defuzzify(act_ki, self.delta_ki_list)
        delta_kd = self._defuzzify(act_kd, self.delta_kd_list)

        kp = self.Kp0 + delta_kp
        ki = self.Ki0 + delta_ki
        kd = self.Kd0 + delta_kd
        kp = np.clip(kp, 0.1, 5.0)
        ki = np.clip(ki, 0.0, 1.0)
        kd = np.clip(kd, 0.0, 5.0)

        u = kp * e_c + ki * ie_c + kd * ec_c
        return np.clip(u, -1.0, 1.0)


# ==================== 加密版 TASE2026 模糊PID（三维分区二次多项式） ====================
class EncryptedTase2026FuzzyController:
    """
    加密版 TASE2026 模糊PID
    纵向：三维分区二次多项式逼近 (e, de, I) → throttle
    横向：Stanley（明文）
    积分器：滑动窗口长度20（运行时的积分值使用滑动窗口计算）
    """
    def __init__(self, waypoints):
        self._current_x = 0.0
        self._current_y = 0.0
        self._current_yaw = 0.0
        self._current_speed = 0.0
        self._desired_speed = 0.0
        self.throttle = 0.0
        self.steer = 0.0
        self._waypoints = waypoints
        self._conv_rad_to_steer = 180.0 / 70.0 / np.pi
        self._pi = np.pi
        self._2pi = 2.0 * np.pi

        # ---------- CKKS 上下文（与您的加密模糊PID一致） ----------
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        self.context.generate_galois_keys()
        self.context.global_scale = 2 ** 40

        # ---------- 三维分区参数（与训练域对齐） ----------
        self.e_bins = [-10, -6, -3, 0, 3, 6, 10]          # 8 个 e 区间
        self.de_bins = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]  # 8 个 de 区间
        self.ie_bins = [-10.0, -5.0, 0.0, 5.0, 10.0]        # 6 个积分区间
        self.num_regions = (len(self.e_bins)+1) * (len(self.de_bins)+1) * (len(self.ie_bins)+1)
        self.e_range = (-15, 15)
        self.de_range = (-3, 3)
        self.ie_range = (-20, 20)

        # 训练多项式系数
        self.local_coeffs = self._fit_all_regions()
        print(f"✅ 加密 TASE2026 模糊PID三维分区多项式已就绪（{self.num_regions} 个区域）")

        # 运行时滑动窗口积分器（长度20）
        self.ie_buffer = deque(maxlen=20)
        self._prev_error = 0.0
        self._e_buffer = []   # 用于计算 de

    def _get_integral(self):
        ie = sum(self.ie_buffer) * DT
        return np.clip(ie, self.ie_range[0], self.ie_range[1])

    # ---------- 训练数据生成 ----------
    def _fit_all_regions(self):
        plain_ref = Tase2026FuzzyTrain()
        coeffs_list = []
        e_edges = [-np.inf] + self.e_bins + [np.inf]
        de_edges = [-np.inf] + self.de_bins + [np.inf]
        ie_edges = [-np.inf] + self.ie_bins + [np.inf]

        e_samples, de_samples, ie_samples = 8, 6, 5

        for i in range(len(e_edges) - 1):
            e_min = max(self.e_range[0], e_edges[i]); e_max = min(self.e_range[1], e_edges[i+1])
            if e_max <= e_min: e_max = e_min + 0.1
            for j in range(len(de_edges) - 1):
                de_min = max(self.de_range[0], de_edges[j]); de_max = min(self.de_range[1], de_edges[j+1])
                if de_max <= de_min: de_max = de_min + 0.1
                for k in range(len(ie_edges) - 1):
                    ie_min = max(self.ie_range[0], ie_edges[k]); ie_max = min(self.ie_range[1], ie_edges[k+1])
                    if ie_max <= ie_min: ie_max = ie_min + 0.1
                    e_vals = np.linspace(e_min, e_max, e_samples)
                    de_vals = np.linspace(de_min, de_max, de_samples)
                    ie_vals = np.linspace(ie_min, ie_max, ie_samples)
                    X, y = [], []
                    for e, de, ie in product(e_vals, de_vals, ie_vals):
                        X.append([1.0, e, de, ie,
                                  e**2, e*de, e*ie,
                                  de**2, de*ie, ie**2])
                        y.append(plain_ref.compute_throttle_with_integral(e, de, ie))
                    X = np.array(X)
                    y = np.array(y)
                    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                    coeffs_list.append(coeffs)
        return coeffs_list

    def _region_index(self, e, de, ie):
        e_idx = 0
        for i, b in enumerate(self.e_bins):
            if e >= b: e_idx = i + 1
        de_idx = 0
        for j, b in enumerate(self.de_bins):
            if de >= b: de_idx = j + 1
        ie_idx = 0
        for k, b in enumerate(self.ie_bins):
            if ie >= b: ie_idx = k + 1
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

    def _compute_longitudinal_throttle_encrypted(self, speed_error, delta_error, integral):
        # 车端明文 clamp 到训练域
        e_c = float(np.clip(speed_error, self.e_range[0], self.e_range[1]))
        de_c = float(np.clip(delta_error, self.de_range[0], self.de_range[1]))
        ie_c = float(np.clip(integral, self.ie_range[0], self.ie_range[1]))

        idx = self._region_index(e_c, de_c, ie_c)
        coeffs = self.local_coeffs[idx]
        enc_e = ts.ckks_vector(self.context, [e_c])
        enc_de = ts.ckks_vector(self.context, [de_c])
        enc_ie = ts.ckks_vector(self.context, [ie_c])
        enc_throttle = self._polyval_encrypted(enc_e, enc_de, enc_ie, coeffs)
        dec = enc_throttle.decrypt()
        throttle = dec[0] if dec else 0.0
        return np.clip(throttle, -1.0, 1.0)

    # ---------- 公共接口 ----------
    def update_values(self, x, y, yaw, speed):
        self._current_x = x
        self._current_y = y
        self._current_yaw = yaw
        self._current_speed = speed

    def update_desired_speed(self):
        diff = self._waypoints[:, :2] - np.array([self._current_x, self._current_y])
        dists = np.sqrt((diff**2).sum(axis=1))
        min_idx = int(np.argmin(dists))
        if min_idx < len(self._waypoints) - 1:
            self._desired_speed = self._waypoints[min_idx, 2]
        else:
            self._desired_speed = self._waypoints[-1, 2]

    def update_waypoints(self, new_waypoints):
        self._waypoints = np.array(new_waypoints)

    def update_controls(self):
        self.update_desired_speed()
        speed_error = self._desired_speed - self._current_speed

        # 更新滑动窗口积分器
        self.ie_buffer.append(speed_error)
        integral = self._get_integral()

        # 计算误差变化率
        self._e_buffer.append(speed_error)
        if len(self._e_buffer) >= 2:
            delta_error = (self._e_buffer[-1] - self._e_buffer[-2]) / DT
        else:
            delta_error = 0.0

        self.throttle = self._compute_longitudinal_throttle_encrypted(speed_error, delta_error, integral)

        # 横向控制（Stanley）
        self._lateral_control()

    def _lateral_control(self):
        """与明文版相同的 Stanley 横向控制"""
        x = self._current_x
        y = self._current_y
        yaw = self._current_yaw
        v = self._current_speed
        waypoints = self._waypoints
        k_e, k_v = 0.3, 20

        yaw_path = np.arctan2(waypoints[-1][1] - waypoints[0][1],
                              waypoints[-1][0] - waypoints[0][0])
        yaw_diff = yaw_path - yaw
        if yaw_diff > np.pi: yaw_diff -= 2 * np.pi
        if yaw_diff < -np.pi: yaw_diff += 2 * np.pi

        current_xy = np.array([x, y])
        crosstrack_error = np.min(np.sum((current_xy - waypoints[:, :2])**2, axis=1))
        yaw_cross_track = np.arctan2(y - waypoints[0][1], x - waypoints[0][0])
        yaw_path2ct = yaw_path - yaw_cross_track
        if yaw_path2ct > np.pi: yaw_path2ct -= 2 * np.pi
        if yaw_path2ct < -np.pi: yaw_path2ct += 2 * np.pi
        if yaw_path2ct > 0:
            crosstrack_error = abs(crosstrack_error)
        else:
            crosstrack_error = -abs(crosstrack_error)

        yaw_diff_crosstrack = np.arctan(k_e * crosstrack_error / (k_v + v))
        steer_expect = yaw_diff + yaw_diff_crosstrack
        if steer_expect > np.pi: steer_expect -= 2 * np.pi
        if steer_expect < -np.pi: steer_expect += 2 * np.pi
        steer_expect = np.clip(steer_expect, -1.22, 1.22)

        self.steer = steering_radians(self._current_x, self._current_y, self._current_yaw, self._current_speed, self._waypoints)


# ==================== 简单测试 ====================
if __name__ == "__main__":
    waypts = np.array([[0, 0, 10], [10, 0, 10], [20, 5, 8], [30, 10, 6]])
    ctrl_plain = Tase2026FuzzyController(waypts)
    ctrl_plain.update_values(5, 0.5, 0.1, 8)
    ctrl_plain.update_controls()
    print(f"明文 throttle = {ctrl_plain.throttle:.4f}, steer = {ctrl_plain.steer:.4f}")

    ctrl_enc = EncryptedTase2026FuzzyController(waypts)
    ctrl_enc.update_values(5, 0.5, 0.1, 8)
    ctrl_enc.update_controls()
    print(f"加密 throttle = {ctrl_enc.throttle:.4f}, steer = {ctrl_enc.steer:.4f}")
    print("✅ 测试完成")