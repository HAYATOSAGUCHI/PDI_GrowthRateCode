# PDI_GrowthRateCode

#シナリオ１　シンプル膨張を使った断熱膨張シナリオ beta=0.001ver　完成版

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import matplotlib.ticker as mticker

# ──────────────────────────── ここを追加 ───────────────────────────
plt.rcParams.update({
    'font.size'        : 26,   # すべての基本フォントサイズ
    'axes.titlesize'   : 26,   # タイトル
    'axes.labelsize'   : 26,   # 軸ラベル
    'xtick.labelsize'  : 24,   # x 目盛ラベル
    'ytick.labelsize'  : 24,   # y 目盛ラベル
    'legend.fontsize'  : 24,   # 凡例
    'figure.titlesize' : 26    # Figure 直下の suptitle など
})
# ────────────────────────────────────────────────────────────────

# ===============================
# 1. モデル（CGL, adiabatic, WKB）の設定
# ===============================
xi_0 = 10
beta_parallel_0 = 0.001/7
B_perp_squared_0 = 0.001

a = 2
b = -2
c = 0.92
d = 2/3

def calculate_max_imaginary_part_CGL(r_ratio, xi_0, beta_parallel_0):
    # r_ratio によりパラメータ更新
    beta_parallel = beta_parallel_0 * r_ratio**(a)
    xi = xi_0 * r_ratio**(b)
    B_perp_squared = B_perp_squared_0 * r_ratio**(c)
    tilde_beta = beta_parallel * (3.0 / 2.0) / (1.0 + B_perp_squared + beta_parallel/2.0*(xi-1.0))

    def newton_method(omega_hat, k_hat):
        f = ((omega_hat**2 - tilde_beta * k_hat**2 * (1.0 + B_perp_squared * xi / 3.0)) *
             ((omega_hat - k_hat) * ((omega_hat + k_hat)**2 - 4.0) +
              tilde_beta * B_perp_squared * (xi - 4.0) / (3.0 * (1.0 + B_perp_squared)) *
              ((k_hat**2 + 1.0) * omega_hat + k_hat * (k_hat**2 - 3.0))) -
             (B_perp_squared * k_hat**2 * (1.0 - tilde_beta * (3.0 - xi - B_perp_squared) / (3.0 * (1.0 + B_perp_squared))) *
              ((omega_hat**3 + omega_hat**2 * k_hat - 3.0 * omega_hat + k_hat) -
               tilde_beta * (3.0 - xi) / 3.0 * ((k_hat**2 + 1.0) * omega_hat + k_hat * (k_hat**2 - 3.0)))))
        df = (2 * omega_hat * ((omega_hat - k_hat) * ((omega_hat + k_hat)**2 - 4.0) +
                             tilde_beta * B_perp_squared * (xi - 4.0) / (3.0 * (1.0 + B_perp_squared)) *
                             ((k_hat**2 + 1.0) * omega_hat + k_hat * (k_hat**2 - 3.0)))
            + (omega_hat**2 - tilde_beta * k_hat**2 * (1.0 + B_perp_squared * xi / 3.0)) *
            (((omega_hat + k_hat)**2 - 4.0) + (omega_hat - k_hat) * 2 * (omega_hat + k_hat))
            + tilde_beta * B_perp_squared * (xi - 4.0) / (3.0 * (1.0 + B_perp_squared)) * (k_hat**2 + 1.0)
            - B_perp_squared * k_hat**2 * (1.0 - tilde_beta * (3.0 - xi - B_perp_squared) / (3.0 * (1.0 + B_perp_squared))) *
            (3 * omega_hat**2 + 2 * omega_hat * k_hat - 3.0)
            - (B_perp_squared * k_hat**2 * (-tilde_beta * (3.0 - xi) / 3.0)) * (k_hat**2 + 1.0) )
        if np.abs(df) < 1e-10:
            return omega_hat
        return omega_hat - f / df

    def root_of_func(initial_guesses, k_hat):
        roots = []
        for omega_hat in initial_guesses:
            for _ in range(1000):
                omega_hat_new = newton_method(omega_hat, k_hat)
                if np.abs(omega_hat_new - omega_hat) < 1e-10:
                    if all(np.abs(omega_hat_new - root) > 1e-10 for root in roots):
                        roots.append(omega_hat_new)
                    break
                omega_hat = omega_hat_new
        return roots

    k_hat_values = np.linspace(0.95, 2.5, 1000)
    initial_guesses = [0.1+0.2j, 2+0.5j, -5+2j, -5-8j, 7-3j, 1+1j, 2+0.1j, 2-0.1j,
                       -2+2j, -2-2j, 3+3j, 3-3j, -4+0.5j, 4+0.5j, 0+3j, 0-3j,
                       1+2j, 1-2j, -1+0.5j, -1-0.5j, 0.5+1j, 0.5-1j, -0.5+0.2j,
                       -0.5-0.2j, 3.5+4j, 3.5-4j, -3.5+4j, -3.5-4j,
                       2.15+0.1j, 2.20+0.1j, 2.25+0.1j, 2.1+0.1j, 2.15+0.07j, 2.15+0.04j, 2.15+0.01j, -4-4j,
                       6+2j, 6-2j, -6+2j, -6-2j, 0.1+0.3j, 0.1-0.3j, -0.1+0.3j, -0.1-0.3j]
    omega_hats_imag = []
    for k_hat in k_hat_values:
        roots = root_of_func(initial_guesses, k_hat)
        for root in roots:
            omega_hats_imag.append(root.imag)
    return np.max(omega_hats_imag)

def calculate_firehose_condition(r_ratios, xi_0, beta_parallel_0, B_perp_squared_0):
    xi_values = xi_0 * r_ratios**(b)
    beta_parallel_values = beta_parallel_0 * r_ratios**(a)
    B_perp_squared_values = B_perp_squared_0 * r_ratios**(c)
    condition = 1 + 0.5 * beta_parallel_values * (xi_values - 1) / (1 + B_perp_squared_values)
    return condition

# ===============================
# 2. モデル（等方的: B,ρ ~ R⁻², Tは pρ^(-5/3)=const, adiabatic, WKB）の設定
# ===============================
beta_0 = 0.001
def calculate_max_imaginary_part_isotropic(args):
    r_ratio, B_perp_squared_0 = args
    beta = beta_0 * r_ratio**(d)
    B_perp_squared = B_perp_squared_0 * r_ratio
    def newton_method(omega_hat, k_hat):
        f = (omega_hat - k_hat) * (omega_hat**2 - beta * k_hat**2) * ((omega_hat + k_hat)**2 - 4) \
            - (B_perp_squared * k_hat**2) * (omega_hat**3 + k_hat * omega_hat**2 - 3 * omega_hat + k_hat)
        df = (3 * B_perp_squared * k_hat**2 + beta * k_hat**4 + 4 * beta * k_hat**2 +
              4 * k_hat * omega_hat**3 + 5 * omega_hat**4 +
              3 * omega_hat**2 * (-B_perp_squared * k_hat**2 - beta * k_hat**2 - k_hat**2 - 4) +
              2 * omega_hat * (-B_perp_squared * k_hat**3 - beta * k_hat**3 - k_hat**3 + 4 * k_hat))
        if np.abs(df) < 1e-10:
            return omega_hat
        return omega_hat - f / df
    def root_of_func(initial_guesses, k_hat):
        roots = []
        for omega_hat in initial_guesses:
            for _ in range(300):
                omega_hat_new = newton_method(omega_hat, k_hat)
                if np.abs(omega_hat_new - omega_hat) < 1e-10:
                    if all(np.abs(omega_hat_new - root) > 1e-10 for root in roots):
                        roots.append(omega_hat_new)
                    break
                omega_hat = omega_hat_new
        return roots
    k_hat_values = np.linspace(0.85, 2.5, 1000)
    initial_guesses = [0.1+0.2j, 2+0.5j, -5+2j, -5-8j, 7-3j, 1+1j, 2+0.1j, 2-0.1j,
                       -2+2j, -2-2j, 3+3j, 3-3j, -4+0.5j, 4+0.5j, 0+3j, 0-3j,
                       1+2j, 1-2j, -1+0.5j, -1-0.5j, 0.5+1j, 0.5-1j, -0.5+0.2j,
                       -0.5-0.2j, 3.5+4j, 3.5-4j, -3.5+4j, -3.5-4j,
                       2.15+0.1j, 2.20+0.1j, 2.25+0.1j, 2.1+0.1j, 2.15+0.07j, 2.15+0.04j, 2.15+0.01j, -4-4j,
                       6+2j, 6-2j, -6+2j, -6-2j, 0.1+0.3j, 0.1-0.3j, -0.1+0.3j, -0.1-0.3j]
    omega_hats_imag = []
    for k_hat in k_hat_values:
        roots = root_of_func(initial_guesses, k_hat)
        for root in roots:
            omega_hats_imag.append(root.imag)
    return np.max(omega_hats_imag)

# ===============================
# 3. 各 r_ratio での計算とプロット
# ===============================
r_ratios = np.linspace(1, 30, 90)

# (A) Firehose 条件（CGLモデル）の計算
firehose_condition = calculate_firehose_condition(r_ratios, xi_0, beta_parallel_0, B_perp_squared_0)
# Firehose 条件が 0 未満となる最初の r_ratio インデックス（条件成立しなくなるところ）
cutoff_index = np.argmax(firehose_condition < 0) if np.any(firehose_condition < 0) else len(r_ratios)
r_ratios_cutoff = r_ratios[:cutoff_index]

# (B) CGLモデルの最大虚部 γₘₐₓ/ω₀ の計算（並列処理）
with Pool() as pool:
    max_imaginary_parts_CGL = pool.starmap(
        calculate_max_imaginary_part_CGL,
        [(r, xi_0, beta_parallel_0) for r in r_ratios_cutoff]
    )

# (C) 等方モデルの最大虚部 γₘₐₓ/ω₀ の計算（並列処理）
with Pool() as pool:
    max_imaginary_parts_iso = pool.map(
        calculate_max_imaginary_part_isotropic,
        [(r, B_perp_squared_0) for r in r_ratios]
    )

# =========================================================
# 連結プロット：上段を下段より少し大きく
# =========================================================
fig, axes = plt.subplots(
    nrows=2, ncols=1, sharex=True, figsize=(15, 10),
    gridspec_kw={
        'hspace': 0.05,           # パネル間の縦スペース
        'height_ratios': [1.7, 1] # ← ここで比率を指定（[上段, 下段]）
    })

# -------------------------
# (1) 上段: γmax / ω0 & Firehose
# -------------------------
ax1  = axes[0]
ax1_r = ax1.twinx()

ax1.plot(r_ratios_cutoff, max_imaginary_parts_CGL,
         color='tab:red', lw=2, label=r"$\gamma_{max}/\omega_{0}$ (CGL)")
ax1.plot(r_ratios, max_imaginary_parts_iso,
         color='tab:green', lw=2, label=r"$\gamma_{max}/\omega_{0}$ (isotropic)")
ax1_r.plot(r_ratios, firehose_condition,
           color='tab:blue', lw=2, label="Discriminant value")

ax1.set_ylabel(r"$\gamma_{max}/\omega_{0}$")
ax1.tick_params(axis='y')
ax1.grid(True, alpha=0.3)
ax1_r.set_ylabel("Discriminant value",color='tab:blue')
ax1_r.tick_params(axis='y',colors='tab:blue')
ax1_r.set_ylim(-1, 1.5)

# 上段凡例（右外）
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_r.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,
           loc='center left', bbox_to_anchor=(1.20, 0.5),
            frameon=False)

# -------------------------
# (2) 下段: β∥, β, ξ, B⊥²
# -------------------------
ax2 = axes[1]

beta_parallel_vals = beta_parallel_0 * r_ratios**(a)
beta_iso_vals      = beta_0 * r_ratios**(d)
xi_vals            = xi_0 * r_ratios**(b)
B_perp_vals        = B_perp_squared_0 * r_ratios**(c)

ax2.plot(r_ratios, beta_iso_vals,      lw=2, color='tab:brown',
            label=r'$\beta$ (isotropic)')
ax2.plot(r_ratios, beta_parallel_vals, lw=2, color='tab:purple',
            label=r'$\beta_{\parallel}$ (CGL)')
ax2.plot(r_ratios, xi_vals,            lw=2, color='tab:orange',
            label=r'$\xi=T_{\perp}/T_{\parallel}$')
ax2.plot(r_ratios, B_perp_vals,        lw=2, color='tab:cyan',
            label=r'$\hat{B}_{\perp}^{2}$')

ax2.set_yscale('log')
ax2.set_xlabel(r'$R/R_{0}$')
ax2.set_ylabel(r'$\beta,\;\beta_{\parallel},\;\xi,\;\hat{B}_{\perp}^{2}$')
ax2.set_ylim(1e-5, 1e2)
ax2.set_yticks([1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2])
from matplotlib.ticker import LogLocator
# ───── log‐grid 追加ここから ─────
ax2.yaxis.set_minor_locator(LogLocator(base=10.0,          # 10 のべき
                                       subs=np.arange(1, 10)*0.1,  # 1–9 の副目盛り
                                       numticks=100))
ax2.grid(True,  which='major', alpha=0.6)  # メジャー
ax2.grid(True,  which='minor', axis='y', alpha=0.3)  # マイナー
# ───── ここまで ─────
ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

# -------------------------
# 図全体レイアウト
# -------------------------
plt.subplots_adjust(right=0.8)  # 右 20 % を凡例スペースに
plt.show()

# ===============================
# 4. 各 r_ratio に対するデータセット (90×3) の作成と出力
# ===============================
# 出力するデータセットの各行は [max_imaginary_CGL, max_imaginary_isotropic, firehose_condition] となる
# ※ CGL の値は、firehose 条件が正の範囲 (r < r_ratios_cutoff) では計算済み。条件が成立しない r_ratio には NaN を補完

# 90 行のデータセットを作成
dataset = np.empty((len(r_ratios), 3))
dataset[:, 1] = max_imaginary_parts_iso          # 等方モデルの値は全 r_ratio に対してある
dataset[:, 2] = firehose_condition               # Firehose 条件

# CGL の値は、cutoff_index まで計算されているので、それ以降は NaN とする
dataset[:cutoff_index, 0] = max_imaginary_parts_CGL
dataset[cutoff_index:, 0] = np.nan

# dataset の各行に対応する r_ratio の値も表示したい場合は、別途 r_ratios を利用するか、カラムとして結合する
# 例: [r_ratio, max_imaginary_CGL, max_imaginary_isotropic, firehose_condition] の 90×4 のデータセット
# 今回は 90×3 (各モデルの値のみ) を出力する

# データセットの出力（例：標準出力に表示）
print("行: r_ratio のインデックス (1～90)")
print("列: [γₘₐₓ/ω₀ (CGL), γₘₐₓ/ω₀ (isotropic), Firehose 条件]")
print(dataset)

# -------------------------
# 追加: 下段プロット用データも含めて保存
# -------------------------
# 8 列: r, γCGL, γiso, Firehose, β∥, β, ξ, B⊥²
full_dataset = np.column_stack([
    r_ratios,
    dataset,                   # 既存の 3 列
    beta_parallel_vals,
    beta_iso_vals,
    xi_vals,
    B_perp_vals
])

# ヘッダー行を列名に合わせて作成
header = ("r_ratio, gamma_CGL, gamma_iso, firehose, "
          "beta_parallel, beta_iso, xi, B_perp_squared")

# 標準出力で確認
print("\n--- full_dataset preview ---")
print(full_dataset[:5])   # 先頭 5 行だけ表示

# CSV 保存
np.savetxt("Fig1a_dataset_full.csv", full_dataset,
           delimiter=",", header=header, comments="")
print(">> Fig1a_dataset_full.csv を保存しました")

#シナリオ１　シンプル膨張を使った断熱膨張シナリオ beta=0.01ver　完成版

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import matplotlib.ticker as mticker

# ===============================
# 1. モデル（CGL, adiabatic, WKB）の設定
# ===============================
xi_0 = 10
beta_parallel_0 = 0.01/7
B_perp_squared_0 = 0.001

a = 2
b = -2
c = 0.92
d = 2/3

def calculate_max_imaginary_part_CGL(r_ratio, xi_0, beta_parallel_0):
    # r_ratio によりパラメータ更新
    beta_parallel = beta_parallel_0 * r_ratio**(a)
    xi = xi_0 * r_ratio**(b)
    B_perp_squared = B_perp_squared_0 * r_ratio**(c)
    tilde_beta = beta_parallel * (3.0 / 2.0) / (1.0 + B_perp_squared + beta_parallel/2.0*(xi-1.0))

    def newton_method(omega_hat, k_hat):
        f = ((omega_hat**2 - tilde_beta * k_hat**2 * (1.0 + B_perp_squared * xi / 3.0)) *
             ((omega_hat - k_hat) * ((omega_hat + k_hat)**2 - 4.0) +
              tilde_beta * B_perp_squared * (xi - 4.0) / (3.0 * (1.0 + B_perp_squared)) *
              ((k_hat**2 + 1.0) * omega_hat + k_hat * (k_hat**2 - 3.0))) -
             (B_perp_squared * k_hat**2 * (1.0 - tilde_beta * (3.0 - xi - B_perp_squared) / (3.0 * (1.0 + B_perp_squared))) *
              ((omega_hat**3 + omega_hat**2 * k_hat - 3.0 * omega_hat + k_hat) -
               tilde_beta * (3.0 - xi) / 3.0 * ((k_hat**2 + 1.0) * omega_hat + k_hat * (k_hat**2 - 3.0)))))
        df = (2 * omega_hat * ((omega_hat - k_hat) * ((omega_hat + k_hat)**2 - 4.0) +
                             tilde_beta * B_perp_squared * (xi - 4.0) / (3.0 * (1.0 + B_perp_squared)) *
                             ((k_hat**2 + 1.0) * omega_hat + k_hat * (k_hat**2 - 3.0)))
            + (omega_hat**2 - tilde_beta * k_hat**2 * (1.0 + B_perp_squared * xi / 3.0)) *
            (((omega_hat + k_hat)**2 - 4.0) + (omega_hat - k_hat) * 2 * (omega_hat + k_hat))
            + tilde_beta * B_perp_squared * (xi - 4.0) / (3.0 * (1.0 + B_perp_squared)) * (k_hat**2 + 1.0)
            - B_perp_squared * k_hat**2 * (1.0 - tilde_beta * (3.0 - xi - B_perp_squared) / (3.0 * (1.0 + B_perp_squared))) *
            (3 * omega_hat**2 + 2 * omega_hat * k_hat - 3.0)
            - (B_perp_squared * k_hat**2 * (-tilde_beta * (3.0 - xi) / 3.0)) * (k_hat**2 + 1.0) )
        if np.abs(df) < 1e-10:
            return omega_hat
        return omega_hat - f / df

    def root_of_func(initial_guesses, k_hat):
        roots = []
        for omega_hat in initial_guesses:
            for _ in range(1000):
                omega_hat_new = newton_method(omega_hat, k_hat)
                if np.abs(omega_hat_new - omega_hat) < 1e-10:
                    if all(np.abs(omega_hat_new - root) > 1e-10 for root in roots):
                        roots.append(omega_hat_new)
                    break
                omega_hat = omega_hat_new
        return roots

    k_hat_values = np.linspace(0.95, 2.5, 1000)
    initial_guesses = [0.1+0.2j, 2+0.5j, -5+2j, -5-8j, 7-3j, 1+1j, 2+0.1j, 2-0.1j,
                       -2+2j, -2-2j, 3+3j, 3-3j, -4+0.5j, 4+0.5j, 0+3j, 0-3j,
                       1+2j, 1-2j, -1+0.5j, -1-0.5j, 0.5+1j, 0.5-1j, -0.5+0.2j,
                       -0.5-0.2j, 3.5+4j, 3.5-4j, -3.5+4j, -3.5-4j,
                       2.15+0.1j, 2.20+0.1j, 2.25+0.1j, 2.1+0.1j, 2.15+0.07j, 2.15+0.04j, 2.15+0.01j, -4-4j,
                       6+2j, 6-2j, -6+2j, -6-2j, 0.1+0.3j, 0.1-0.3j, -0.1+0.3j, -0.1-0.3j]
    omega_hats_imag = []
    for k_hat in k_hat_values:
        roots = root_of_func(initial_guesses, k_hat)
        for root in roots:
            omega_hats_imag.append(root.imag)
    return np.max(omega_hats_imag)

def calculate_firehose_condition(r_ratios, xi_0, beta_parallel_0, B_perp_squared_0):
    xi_values = xi_0 * r_ratios**(b)
    beta_parallel_values = beta_parallel_0 * r_ratios**(a)
    B_perp_squared_values = B_perp_squared_0 * r_ratios**(c)
    condition = 1 + 0.5 * beta_parallel_values * (xi_values - 1) / (1 + B_perp_squared_values)
    return condition

# ===============================
# 2. モデル（等方的: B,ρ ~ R⁻², Tは pρ^(-5/3)=const, adiabatic, WKB）の設定
# ===============================
beta_0 = 0.01
def calculate_max_imaginary_part_isotropic(args):
    r_ratio, B_perp_squared_0 = args
    beta = beta_0 * r_ratio**(d)
    B_perp_squared = B_perp_squared_0 * r_ratio
    def newton_method(omega_hat, k_hat):
        f = (omega_hat - k_hat) * (omega_hat**2 - beta * k_hat**2) * ((omega_hat + k_hat)**2 - 4) \
            - (B_perp_squared * k_hat**2) * (omega_hat**3 + k_hat * omega_hat**2 - 3 * omega_hat + k_hat)
        df = (3 * B_perp_squared * k_hat**2 + beta * k_hat**4 + 4 * beta * k_hat**2 +
              4 * k_hat * omega_hat**3 + 5 * omega_hat**4 +
              3 * omega_hat**2 * (-B_perp_squared * k_hat**2 - beta * k_hat**2 - k_hat**2 - 4) +
              2 * omega_hat * (-B_perp_squared * k_hat**3 - beta * k_hat**3 - k_hat**3 + 4 * k_hat))
        if np.abs(df) < 1e-10:
            return omega_hat
        return omega_hat - f / df
    def root_of_func(initial_guesses, k_hat):
        roots = []
        for omega_hat in initial_guesses:
            for _ in range(300):
                omega_hat_new = newton_method(omega_hat, k_hat)
                if np.abs(omega_hat_new - omega_hat) < 1e-10:
                    if all(np.abs(omega_hat_new - root) > 1e-10 for root in roots):
                        roots.append(omega_hat_new)
                    break
                omega_hat = omega_hat_new
        return roots
    k_hat_values = np.linspace(0.85, 2.5, 1000)
    initial_guesses = [0.1+0.2j, 2+0.5j, -5+2j, -5-8j, 7-3j, 1+1j, 2+0.1j, 2-0.1j,
                       -2+2j, -2-2j, 3+3j, 3-3j, -4+0.5j, 4+0.5j, 0+3j, 0-3j,
                       1+2j, 1-2j, -1+0.5j, -1-0.5j, 0.5+1j, 0.5-1j, -0.5+0.2j,
                       -0.5-0.2j, 3.5+4j, 3.5-4j, -3.5+4j, -3.5-4j,
                       2.15+0.1j, 2.20+0.1j, 2.25+0.1j, 2.1+0.1j, 2.15+0.07j, 2.15+0.04j, 2.15+0.01j, -4-4j,
                       6+2j, 6-2j, -6+2j, -6-2j, 0.1+0.3j, 0.1-0.3j, -0.1+0.3j, -0.1-0.3j]
    omega_hats_imag = []
    for k_hat in k_hat_values:
        roots = root_of_func(initial_guesses, k_hat)
        for root in roots:
            omega_hats_imag.append(root.imag)
    return np.max(omega_hats_imag)

# ===============================
# 3. 各 r_ratio での計算とプロット
# ===============================
r_ratios = np.linspace(1, 30, 90)

# (A) Firehose 条件（CGLモデル）の計算
firehose_condition = calculate_firehose_condition(r_ratios, xi_0, beta_parallel_0, B_perp_squared_0)
# Firehose 条件が 0 未満となる最初の r_ratio インデックス（条件成立しなくなるところ）
cutoff_index = np.argmax(firehose_condition < 0) if np.any(firehose_condition < 0) else len(r_ratios)
r_ratios_cutoff = r_ratios[:cutoff_index]

# (B) CGLモデルの最大虚部 γₘₐₓ/ω₀ の計算（並列処理）
with Pool() as pool:
    max_imaginary_parts_CGL = pool.starmap(
        calculate_max_imaginary_part_CGL,
        [(r, xi_0, beta_parallel_0) for r in r_ratios_cutoff]
    )

# (C) 等方モデルの最大虚部 γₘₐₓ/ω₀ の計算（並列処理）
with Pool() as pool:
    max_imaginary_parts_iso = pool.map(
        calculate_max_imaginary_part_isotropic,
        [(r, B_perp_squared_0) for r in r_ratios]
    )

# =========================================================
# 連結プロット：上段を下段より少し大きく
# =========================================================
fig, axes = plt.subplots(
    nrows=2, ncols=1, sharex=True, figsize=(15, 10),
    gridspec_kw={
        'hspace': 0.05,           # パネル間の縦スペース
        'height_ratios': [1.7, 1] # ← ここで比率を指定（[上段, 下段]）
    })

# -------------------------
# (1) 上段: γmax / ω0 & Firehose
# -------------------------
ax1  = axes[0]
ax1_r = ax1.twinx()

ax1.plot(r_ratios_cutoff, max_imaginary_parts_CGL,
         color='tab:red', lw=2, label=r"$\gamma_{max}/\omega_{0}$ (CGL)")
ax1.plot(r_ratios, max_imaginary_parts_iso,
         color='tab:green', lw=2, label=r"$\gamma_{max}/\omega_{0}$ (isotropic)")
ax1_r.plot(r_ratios, firehose_condition,
           color='tab:blue', lw=2, label="Discriminant value")

ax1.set_ylabel(r"$\gamma_{max}/\omega_{0}$", fontsize=22)
ax1.tick_params(axis='y', labelsize=18)
ax1.grid(True, alpha=0.3)
ax1_r.set_ylabel("Discriminant value", fontsize=22, color='tab:blue')
ax1_r.tick_params(axis='y', labelsize=18, colors='tab:blue')
ax1_r.set_ylim(-1, 1.5)

# 上段凡例（右外）
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_r.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,
           loc='center left', bbox_to_anchor=(1.20, 0.5),
           fontsize=16, frameon=False)

# -------------------------
# (2) 下段: β∥, β, ξ, B⊥²
# -------------------------
ax2 = axes[1]

beta_parallel_vals = beta_parallel_0 * r_ratios**(a)
beta_iso_vals      = beta_0 * r_ratios**(d)
xi_vals            = xi_0 * r_ratios**(b)
B_perp_vals        = B_perp_squared_0 * r_ratios**(c)

ax2.plot(r_ratios, beta_iso_vals,      lw=2, color='tab:brown',
            label=r'$\beta$ (isotropic)')
ax2.plot(r_ratios, beta_parallel_vals, lw=2, color='tab:purple',
            label=r'$\beta_{\parallel}$ (CGL)')
ax2.plot(r_ratios, xi_vals,            lw=2, color='tab:orange',
            label=r'$\xi=T_{\perp}/T_{\parallel}$')
ax2.plot(r_ratios, B_perp_vals,        lw=2, color='tab:cyan',
            label=r'$\hat{B}_{\perp}^{2}$')

ax2.set_yscale('log')
ax2.set_xlabel(r'$R/R_{0}$')
ax2.set_ylabel(r'$\beta,\;\beta_{\parallel},\;\xi,\;\hat{B}_{\perp}^{2}$')
ax2.set_ylim(1e-5, 1e2)
ax2.set_yticks([1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2])
from matplotlib.ticker import LogLocator
# ───── log‐grid 追加ここから ─────
ax2.yaxis.set_minor_locator(LogLocator(base=10.0,          # 10 のべき
                                       subs=np.arange(1, 10)*0.1,  # 1–9 の副目盛り
                                       numticks=100))
ax2.grid(True,  which='major', alpha=0.6)  # メジャー
ax2.grid(True,  which='minor', axis='y', alpha=0.3)  # マイナー
# ───── ここまで ─────
ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

# -------------------------
# 図全体レイアウト
# -------------------------
plt.subplots_adjust(right=0.8)  # 右 20 % を凡例スペースに
plt.show()

# ===============================
# 4. 各 r_ratio に対するデータセット (90×3) の作成と出力
# ===============================
# 出力するデータセットの各行は [max_imaginary_CGL, max_imaginary_isotropic, firehose_condition] となる
# ※ CGL の値は、firehose 条件が正の範囲 (r < r_ratios_cutoff) では計算済み。条件が成立しない r_ratio には NaN を補完

# 90 行のデータセットを作成
dataset = np.empty((len(r_ratios), 3))
dataset[:, 1] = max_imaginary_parts_iso          # 等方モデルの値は全 r_ratio に対してある
dataset[:, 2] = firehose_condition               # Firehose 条件

# CGL の値は、cutoff_index まで計算されているので、それ以降は NaN とする
dataset[:cutoff_index, 0] = max_imaginary_parts_CGL
dataset[cutoff_index:, 0] = np.nan

# dataset の各行に対応する r_ratio の値も表示したい場合は、別途 r_ratios を利用するか、カラムとして結合する
# 例: [r_ratio, max_imaginary_CGL, max_imaginary_isotropic, firehose_condition] の 90×4 のデータセット
# 今回は 90×3 (各モデルの値のみ) を出力する

# データセットの出力（例：標準出力に表示）
print("行: r_ratio のインデックス (1～90)")
print("列: [γₘₐₓ/ω₀ (CGL), γₘₐₓ/ω₀ (isotropic), Firehose 条件]")
print(dataset)

# -------------------------
# 追加: 下段プロット用データも含めて保存
# -------------------------
# 8 列: r, γCGL, γiso, Firehose, β∥, β, ξ, B⊥²
full_dataset = np.column_stack([
    r_ratios,
    dataset,                   # 既存の 3 列
    beta_parallel_vals,
    beta_iso_vals,
    xi_vals,
    B_perp_vals
])

# ヘッダー行を列名に合わせて作成
header = ("r_ratio, gamma_CGL, gamma_iso, firehose, "
          "beta_parallel, beta_iso, xi, B_perp_squared")

# 標準出力で確認
print("\n--- full_dataset preview ---")
print(full_dataset[:5])   # 先頭 5 行だけ表示

# CSV 保存
np.savetxt("Fig1b_dataset_full.csv", full_dataset,
           delimiter=",", header=header, comments="")
print(">> Fig1b_dataset_full.csv を保存しました")

#シナリオ１　シンプル膨張を使った断熱膨張シナリオ beta=0.1ver　完成版

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import matplotlib.ticker as mticker

# ===============================
# 1. モデル（CGL, adiabatic, WKB）の設定
# ===============================
xi_0 = 10
beta_parallel_0 = 0.1/7
B_perp_squared_0 = 0.001

a = 2
b = -2
c = 0.92
d = 2/3

def calculate_max_imaginary_part_CGL(r_ratio, xi_0, beta_parallel_0):
    # r_ratio によりパラメータ更新
    beta_parallel = beta_parallel_0 * r_ratio**(a)
    xi = xi_0 * r_ratio**(b)
    B_perp_squared = B_perp_squared_0 * r_ratio**(c)
    tilde_beta = beta_parallel * (3.0 / 2.0) / (1.0 + B_perp_squared + beta_parallel/2.0*(xi-1.0))

    def newton_method(omega_hat, k_hat):
        f = ((omega_hat**2 - tilde_beta * k_hat**2 * (1.0 + B_perp_squared * xi / 3.0)) *
             ((omega_hat - k_hat) * ((omega_hat + k_hat)**2 - 4.0) +
              tilde_beta * B_perp_squared * (xi - 4.0) / (3.0 * (1.0 + B_perp_squared)) *
              ((k_hat**2 + 1.0) * omega_hat + k_hat * (k_hat**2 - 3.0))) -
             (B_perp_squared * k_hat**2 * (1.0 - tilde_beta * (3.0 - xi - B_perp_squared) / (3.0 * (1.0 + B_perp_squared))) *
              ((omega_hat**3 + omega_hat**2 * k_hat - 3.0 * omega_hat + k_hat) -
               tilde_beta * (3.0 - xi) / 3.0 * ((k_hat**2 + 1.0) * omega_hat + k_hat * (k_hat**2 - 3.0)))))
        df = (2 * omega_hat * ((omega_hat - k_hat) * ((omega_hat + k_hat)**2 - 4.0) +
                             tilde_beta * B_perp_squared * (xi - 4.0) / (3.0 * (1.0 + B_perp_squared)) *
                             ((k_hat**2 + 1.0) * omega_hat + k_hat * (k_hat**2 - 3.0)))
            + (omega_hat**2 - tilde_beta * k_hat**2 * (1.0 + B_perp_squared * xi / 3.0)) *
            (((omega_hat + k_hat)**2 - 4.0) + (omega_hat - k_hat) * 2 * (omega_hat + k_hat))
            + tilde_beta * B_perp_squared * (xi - 4.0) / (3.0 * (1.0 + B_perp_squared)) * (k_hat**2 + 1.0)
            - B_perp_squared * k_hat**2 * (1.0 - tilde_beta * (3.0 - xi - B_perp_squared) / (3.0 * (1.0 + B_perp_squared))) *
            (3 * omega_hat**2 + 2 * omega_hat * k_hat - 3.0)
            - (B_perp_squared * k_hat**2 * (-tilde_beta * (3.0 - xi) / 3.0)) * (k_hat**2 + 1.0) )
        if np.abs(df) < 1e-10:
            return omega_hat
        return omega_hat - f / df

    def root_of_func(initial_guesses, k_hat):
        roots = []
        for omega_hat in initial_guesses:
            for _ in range(1000):
                omega_hat_new = newton_method(omega_hat, k_hat)
                if np.abs(omega_hat_new - omega_hat) < 1e-10:
                    if all(np.abs(omega_hat_new - root) > 1e-10 for root in roots):
                        roots.append(omega_hat_new)
                    break
                omega_hat = omega_hat_new
        return roots

    k_hat_values = np.linspace(0.95, 2.5, 1000)
    initial_guesses = [0.1+0.2j, 2+0.5j, -5+2j, -5-8j, 7-3j, 1+1j, 2+0.1j, 2-0.1j,
                       -2+2j, -2-2j, 3+3j, 3-3j, -4+0.5j, 4+0.5j, 0+3j, 0-3j,
                       1+2j, 1-2j, -1+0.5j, -1-0.5j, 0.5+1j, 0.5-1j, -0.5+0.2j,
                       -0.5-0.2j, 3.5+4j, 3.5-4j, -3.5+4j, -3.5-4j,
                       2.15+0.1j, 2.20+0.1j, 2.25+0.1j, 2.1+0.1j, 2.15+0.07j, 2.15+0.04j, 2.15+0.01j, -4-4j,
                       6+2j, 6-2j, -6+2j, -6-2j, 0.1+0.3j, 0.1-0.3j, -0.1+0.3j, -0.1-0.3j]
    omega_hats_imag = []
    for k_hat in k_hat_values:
        roots = root_of_func(initial_guesses, k_hat)
        for root in roots:
            omega_hats_imag.append(root.imag)
    return np.max(omega_hats_imag)

def calculate_firehose_condition(r_ratios, xi_0, beta_parallel_0, B_perp_squared_0):
    xi_values = xi_0 * r_ratios**(b)
    beta_parallel_values = beta_parallel_0 * r_ratios**(a)
    B_perp_squared_values = B_perp_squared_0 * r_ratios**(c)
    condition = 1 + 0.5 * beta_parallel_values * (xi_values - 1) / (1 + B_perp_squared_values)
    return condition

# ===============================
# 2. モデル（等方的: B,ρ ~ R⁻², Tは pρ^(-5/3)=const, adiabatic, WKB）の設定
# ===============================
beta_0 = 0.1
def calculate_max_imaginary_part_isotropic(args):
    r_ratio, B_perp_squared_0 = args
    beta = beta_0 * r_ratio**(d)
    B_perp_squared = B_perp_squared_0 * r_ratio
    def newton_method(omega_hat, k_hat):
        f = (omega_hat - k_hat) * (omega_hat**2 - beta * k_hat**2) * ((omega_hat + k_hat)**2 - 4) \
            - (B_perp_squared * k_hat**2) * (omega_hat**3 + k_hat * omega_hat**2 - 3 * omega_hat + k_hat)
        df = (3 * B_perp_squared * k_hat**2 + beta * k_hat**4 + 4 * beta * k_hat**2 +
              4 * k_hat * omega_hat**3 + 5 * omega_hat**4 +
              3 * omega_hat**2 * (-B_perp_squared * k_hat**2 - beta * k_hat**2 - k_hat**2 - 4) +
              2 * omega_hat * (-B_perp_squared * k_hat**3 - beta * k_hat**3 - k_hat**3 + 4 * k_hat))
        if np.abs(df) < 1e-10:
            return omega_hat
        return omega_hat - f / df
    def root_of_func(initial_guesses, k_hat):
        roots = []
        for omega_hat in initial_guesses:
            for _ in range(300):
                omega_hat_new = newton_method(omega_hat, k_hat)
                if np.abs(omega_hat_new - omega_hat) < 1e-10:
                    if all(np.abs(omega_hat_new - root) > 1e-10 for root in roots):
                        roots.append(omega_hat_new)
                    break
                omega_hat = omega_hat_new
        return roots
    k_hat_values = np.linspace(0.85, 2.5, 1000)
    initial_guesses = [0.1+0.2j, 2+0.5j, -5+2j, -5-8j, 7-3j, 1+1j, 2+0.1j, 2-0.1j,
                       -2+2j, -2-2j, 3+3j, 3-3j, -4+0.5j, 4+0.5j, 0+3j, 0-3j,
                       1+2j, 1-2j, -1+0.5j, -1-0.5j, 0.5+1j, 0.5-1j, -0.5+0.2j,
                       -0.5-0.2j, 3.5+4j, 3.5-4j, -3.5+4j, -3.5-4j,
                       2.15+0.1j, 2.20+0.1j, 2.25+0.1j, 2.1+0.1j, 2.15+0.07j, 2.15+0.04j, 2.15+0.01j, -4-4j,
                       6+2j, 6-2j, -6+2j, -6-2j, 0.1+0.3j, 0.1-0.3j, -0.1+0.3j, -0.1-0.3j]
    omega_hats_imag = []
    for k_hat in k_hat_values:
        roots = root_of_func(initial_guesses, k_hat)
        for root in roots:
            omega_hats_imag.append(root.imag)
    return np.max(omega_hats_imag)

# ===============================
# 3. 各 r_ratio での計算とプロット
# ===============================
r_ratios = np.linspace(1, 30, 90)

# (A) Firehose 条件（CGLモデル）の計算
firehose_condition = calculate_firehose_condition(r_ratios, xi_0, beta_parallel_0, B_perp_squared_0)
# Firehose 条件が 0 未満となる最初の r_ratio インデックス（条件成立しなくなるところ）
cutoff_index = np.argmax(firehose_condition < 0) if np.any(firehose_condition < 0) else len(r_ratios)
r_ratios_cutoff = r_ratios[:cutoff_index]

# (B) CGLモデルの最大虚部 γₘₐₓ/ω₀ の計算（並列処理）
with Pool() as pool:
    max_imaginary_parts_CGL = pool.starmap(
        calculate_max_imaginary_part_CGL,
        [(r, xi_0, beta_parallel_0) for r in r_ratios_cutoff]
    )

# (C) 等方モデルの最大虚部 γₘₐₓ/ω₀ の計算（並列処理）
with Pool() as pool:
    max_imaginary_parts_iso = pool.map(
        calculate_max_imaginary_part_isotropic,
        [(r, B_perp_squared_0) for r in r_ratios]
    )

# =========================================================
# 連結プロット：上段を下段より少し大きく
# =========================================================
fig, axes = plt.subplots(
    nrows=2, ncols=1, sharex=True, figsize=(15, 10),
    gridspec_kw={
        'hspace': 0.05,           # パネル間の縦スペース
        'height_ratios': [1.7, 1] # ← ここで比率を指定（[上段, 下段]）
    })

# -------------------------
# (1) 上段: γmax / ω0 & Firehose
# -------------------------
ax1  = axes[0]
ax1_r = ax1.twinx()

ax1.plot(r_ratios_cutoff, max_imaginary_parts_CGL,
         color='tab:red', lw=2, label=r"$\gamma_{max}/\omega_{0}$ (CGL)")
ax1.plot(r_ratios, max_imaginary_parts_iso,
         color='tab:green', lw=2, label=r"$\gamma_{max}/\omega_{0}$ (isotropic)")
ax1_r.plot(r_ratios, firehose_condition,
           color='tab:blue', lw=2, label="Discriminant value")

ax1.set_ylabel(r"$\gamma_{max}/\omega_{0}$", fontsize=22)
ax1.tick_params(axis='y', labelsize=18)
ax1.grid(True, alpha=0.3)
ax1_r.set_ylabel("Discriminant value", fontsize=22, color='tab:blue')
ax1_r.tick_params(axis='y', labelsize=18, colors='tab:blue')
ax1_r.set_ylim(-1, 1.5)

# 上段凡例（右外）
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_r.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,
           loc='center left', bbox_to_anchor=(1.20, 0.5),
           fontsize=16, frameon=False)

# -------------------------
# (2) 下段: β∥, β, ξ, B⊥²
# -------------------------
ax2 = axes[1]

beta_parallel_vals = beta_parallel_0 * r_ratios**(a)
beta_iso_vals      = beta_0 * r_ratios**(d)
xi_vals            = xi_0 * r_ratios**(b)
B_perp_vals        = B_perp_squared_0 * r_ratios**(c)

ax2.plot(r_ratios, beta_iso_vals,      lw=2, color='tab:brown',
            label=r'$\beta$ (isotropic)')
ax2.plot(r_ratios, beta_parallel_vals, lw=2, color='tab:purple',
            label=r'$\beta_{\parallel}$ (CGL)')
ax2.plot(r_ratios, xi_vals,            lw=2, color='tab:orange',
            label=r'$\xi=T_{\perp}/T_{\parallel}$')
ax2.plot(r_ratios, B_perp_vals,        lw=2, color='tab:cyan',
            label=r'$\hat{B}_{\perp}^{2}$')

ax2.set_yscale('log')
ax2.set_xlabel(r'$R/R_{0}$')
ax2.set_ylabel(r'$\beta,\;\beta_{\parallel},\;\xi,\;\hat{B}_{\perp}^{2}$')
ax2.set_ylim(1e-5, 1e2)
ax2.set_yticks([1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2])
from matplotlib.ticker import LogLocator
# ───── log‐grid 追加ここから ─────
ax2.yaxis.set_minor_locator(LogLocator(base=10.0,          # 10 のべき
                                       subs=np.arange(1, 10)*0.1,  # 1–9 の副目盛り
                                       numticks=100))
ax2.grid(True,  which='major', alpha=0.6)  # メジャー
ax2.grid(True,  which='minor', axis='y', alpha=0.3)  # マイナー
# ───── ここまで ─────
ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

# -------------------------
# 図全体レイアウト
# -------------------------
plt.subplots_adjust(right=0.8)  # 右 20 % を凡例スペースに
plt.show()

# ===============================
# 4. 各 r_ratio に対するデータセット (90×3) の作成と出力
# ===============================
# 出力するデータセットの各行は [max_imaginary_CGL, max_imaginary_isotropic, firehose_condition] となる
# ※ CGL の値は、firehose 条件が正の範囲 (r < r_ratios_cutoff) では計算済み。条件が成立しない r_ratio には NaN を補完

# 90 行のデータセットを作成
dataset = np.empty((len(r_ratios), 3))
dataset[:, 1] = max_imaginary_parts_iso          # 等方モデルの値は全 r_ratio に対してある
dataset[:, 2] = firehose_condition               # Firehose 条件

# CGL の値は、cutoff_index まで計算されているので、それ以降は NaN とする
dataset[:cutoff_index, 0] = max_imaginary_parts_CGL
dataset[cutoff_index:, 0] = np.nan

# dataset の各行に対応する r_ratio の値も表示したい場合は、別途 r_ratios を利用するか、カラムとして結合する
# 例: [r_ratio, max_imaginary_CGL, max_imaginary_isotropic, firehose_condition] の 90×4 のデータセット
# 今回は 90×3 (各モデルの値のみ) を出力する

# データセットの出力（例：標準出力に表示）
print("行: r_ratio のインデックス (1～90)")
print("列: [γₘₐₓ/ω₀ (CGL), γₘₐₓ/ω₀ (isotropic), Firehose 条件]")
print(dataset)

# -------------------------
# 追加: 下段プロット用データも含めて保存
# -------------------------
# 8 列: r, γCGL, γiso, Firehose, β∥, β, ξ, B⊥²
full_dataset = np.column_stack([
    r_ratios,
    dataset,                   # 既存の 3 列
    beta_parallel_vals,
    beta_iso_vals,
    xi_vals,
    B_perp_vals
])

# ヘッダー行を列名に合わせて作成
header = ("r_ratio, gamma_CGL, gamma_iso, firehose, "
          "beta_parallel, beta_iso, xi, B_perp_squared")

# 標準出力で確認
print("\n--- full_dataset preview ---")
print(full_dataset[:5])   # 先頭 5 行だけ表示

# CSV 保存
np.savetxt("Fig1c_dataset_full.csv", full_dataset,
           delimiter=",", header=header, comments="")
print(">> Fig1c_dataset_full.csv を保存しました")

#シナリオ2　PSPデータに基づいたBとρのスケーリングを使った断熱膨張シナリオ beta=0.001ver　完成版

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import matplotlib.ticker as mticker

# ===============================
# 1. モデル（CGL, adiabatic, WKB）の設定
# ===============================
xi_0 = 10
beta_parallel_0 = 0.001/7
B_perp_squared_0 = 0.001

a = -1.13
b = 0.2
c = 0.92
d = -2.59*(5/3)+3.32

def calculate_max_imaginary_part_CGL(r_ratio, xi_0, beta_parallel_0):
    # r_ratio によりパラメータ更新
    beta_parallel = beta_parallel_0 * r_ratio**(a)
    xi = xi_0 * r_ratio**(b)
    B_perp_squared = B_perp_squared_0 * r_ratio**(c)
    tilde_beta = beta_parallel * (3.0 / 2.0) / (1.0 + B_perp_squared + beta_parallel/2.0*(xi-1.0))

    def newton_method(omega_hat, k_hat):
        f = ((omega_hat**2 - tilde_beta * k_hat**2 * (1.0 + B_perp_squared * xi / 3.0)) *
             ((omega_hat - k_hat) * ((omega_hat + k_hat)**2 - 4.0) +
              tilde_beta * B_perp_squared * (xi - 4.0) / (3.0 * (1.0 + B_perp_squared)) *
              ((k_hat**2 + 1.0) * omega_hat + k_hat * (k_hat**2 - 3.0))) -
             (B_perp_squared * k_hat**2 * (1.0 - tilde_beta * (3.0 - xi - B_perp_squared) / (3.0 * (1.0 + B_perp_squared))) *
              ((omega_hat**3 + omega_hat**2 * k_hat - 3.0 * omega_hat + k_hat) -
               tilde_beta * (3.0 - xi) / 3.0 * ((k_hat**2 + 1.0) * omega_hat + k_hat * (k_hat**2 - 3.0)))))
        df = (2 * omega_hat * ((omega_hat - k_hat) * ((omega_hat + k_hat)**2 - 4.0) +
                             tilde_beta * B_perp_squared * (xi - 4.0) / (3.0 * (1.0 + B_perp_squared)) *
                             ((k_hat**2 + 1.0) * omega_hat + k_hat * (k_hat**2 - 3.0)))
            + (omega_hat**2 - tilde_beta * k_hat**2 * (1.0 + B_perp_squared * xi / 3.0)) *
            (((omega_hat + k_hat)**2 - 4.0) + (omega_hat - k_hat) * 2 * (omega_hat + k_hat))
            + tilde_beta * B_perp_squared * (xi - 4.0) / (3.0 * (1.0 + B_perp_squared)) * (k_hat**2 + 1.0)
            - B_perp_squared * k_hat**2 * (1.0 - tilde_beta * (3.0 - xi - B_perp_squared) / (3.0 * (1.0 + B_perp_squared))) *
            (3 * omega_hat**2 + 2 * omega_hat * k_hat - 3.0)
            - (B_perp_squared * k_hat**2 * (-tilde_beta * (3.0 - xi) / 3.0)) * (k_hat**2 + 1.0) )
        if np.abs(df) < 1e-10:
            return omega_hat
        return omega_hat - f / df

    def root_of_func(initial_guesses, k_hat):
        roots = []
        for omega_hat in initial_guesses:
            for _ in range(1000):
                omega_hat_new = newton_method(omega_hat, k_hat)
                if np.abs(omega_hat_new - omega_hat) < 1e-10:
                    if all(np.abs(omega_hat_new - root) > 1e-10 for root in roots):
                        roots.append(omega_hat_new)
                    break
                omega_hat = omega_hat_new
        return roots

    k_hat_values = np.linspace(0.95, 2.5, 1000)
    initial_guesses = [0.1+0.2j, 2+0.5j, -5+2j, -5-8j, 7-3j, 1+1j, 2+0.1j, 2-0.1j,
                       -2+2j, -2-2j, 3+3j, 3-3j, -4+0.5j, 4+0.5j, 0+3j, 0-3j,
                       1+2j, 1-2j, -1+0.5j, -1-0.5j, 0.5+1j, 0.5-1j, -0.5+0.2j,
                       -0.5-0.2j, 3.5+4j, 3.5-4j, -3.5+4j, -3.5-4j,
                       2.15+0.1j, 2.20+0.1j, 2.25+0.1j, 2.1+0.1j, 2.15+0.07j, 2.15+0.04j, 2.15+0.01j, -4-4j,
                       6+2j, 6-2j, -6+2j, -6-2j, 0.1+0.3j, 0.1-0.3j, -0.1+0.3j, -0.1-0.3j]
    omega_hats_imag = []
    for k_hat in k_hat_values:
        roots = root_of_func(initial_guesses, k_hat)
        for root in roots:
            omega_hats_imag.append(root.imag)
    return np.max(omega_hats_imag)

def calculate_firehose_condition(r_ratios, xi_0, beta_parallel_0, B_perp_squared_0):
    xi_values = xi_0 * r_ratios**(b)
    beta_parallel_values = beta_parallel_0 * r_ratios**(a)
    B_perp_squared_values = B_perp_squared_0 * r_ratios**(c)
    condition = 1 + 0.5 * beta_parallel_values * (xi_values - 1) / (1 + B_perp_squared_values)
    return condition

# ===============================
# 2. モデル（等方的: B,ρ ~ R⁻², Tは pρ^(-5/3)=const, adiabatic, WKB）の設定
# ===============================
beta_0 = 0.001
def calculate_max_imaginary_part_isotropic(args):
    r_ratio, B_perp_squared_0 = args
    beta = beta_0 * r_ratio**(d)
    B_perp_squared = B_perp_squared_0 * r_ratio
    def newton_method(omega_hat, k_hat):
        f = (omega_hat - k_hat) * (omega_hat**2 - beta * k_hat**2) * ((omega_hat + k_hat)**2 - 4) \
            - (B_perp_squared * k_hat**2) * (omega_hat**3 + k_hat * omega_hat**2 - 3 * omega_hat + k_hat)
        df = (3 * B_perp_squared * k_hat**2 + beta * k_hat**4 + 4 * beta * k_hat**2 +
              4 * k_hat * omega_hat**3 + 5 * omega_hat**4 +
              3 * omega_hat**2 * (-B_perp_squared * k_hat**2 - beta * k_hat**2 - k_hat**2 - 4) +
              2 * omega_hat * (-B_perp_squared * k_hat**3 - beta * k_hat**3 - k_hat**3 + 4 * k_hat))
        if np.abs(df) < 1e-10:
            return omega_hat
        return omega_hat - f / df
    def root_of_func(initial_guesses, k_hat):
        roots = []
        for omega_hat in initial_guesses:
            for _ in range(300):
                omega_hat_new = newton_method(omega_hat, k_hat)
                if np.abs(omega_hat_new - omega_hat) < 1e-10:
                    if all(np.abs(omega_hat_new - root) > 1e-10 for root in roots):
                        roots.append(omega_hat_new)
                    break
                omega_hat = omega_hat_new
        return roots
    k_hat_values = np.linspace(0.85, 2.5, 1000)
    initial_guesses = [0.1+0.2j, 2+0.5j, -5+2j, -5-8j, 7-3j, 1+1j, 2+0.1j, 2-0.1j,
                       -2+2j, -2-2j, 3+3j, 3-3j, -4+0.5j, 4+0.5j, 0+3j, 0-3j,
                       1+2j, 1-2j, -1+0.5j, -1-0.5j, 0.5+1j, 0.5-1j, -0.5+0.2j,
                       -0.5-0.2j, 3.5+4j, 3.5-4j, -3.5+4j, -3.5-4j,
                       2.15+0.1j, 2.20+0.1j, 2.25+0.1j, 2.1+0.1j, 2.15+0.07j, 2.15+0.04j, 2.15+0.01j, -4-4j,
                       6+2j, 6-2j, -6+2j, -6-2j, 0.1+0.3j, 0.1-0.3j, -0.1+0.3j, -0.1-0.3j]
    omega_hats_imag = []
    for k_hat in k_hat_values:
        roots = root_of_func(initial_guesses, k_hat)
        for root in roots:
            omega_hats_imag.append(root.imag)
    return np.max(omega_hats_imag)

# ===============================
# 3. 各 r_ratio での計算とプロット
# ===============================
r_ratios = np.linspace(1, 30, 90)

# (A) Firehose 条件（CGLモデル）の計算
firehose_condition = calculate_firehose_condition(r_ratios, xi_0, beta_parallel_0, B_perp_squared_0)
# Firehose 条件が 0 未満となる最初の r_ratio インデックス（条件成立しなくなるところ）
cutoff_index = np.argmax(firehose_condition < 0) if np.any(firehose_condition < 0) else len(r_ratios)
r_ratios_cutoff = r_ratios[:cutoff_index]

# (B) CGLモデルの最大虚部 γₘₐₓ/ω₀ の計算（並列処理）
with Pool() as pool:
    max_imaginary_parts_CGL = pool.starmap(
        calculate_max_imaginary_part_CGL,
        [(r, xi_0, beta_parallel_0) for r in r_ratios_cutoff]
    )

# (C) 等方モデルの最大虚部 γₘₐₓ/ω₀ の計算（並列処理）
with Pool() as pool:
    max_imaginary_parts_iso = pool.map(
        calculate_max_imaginary_part_isotropic,
        [(r, B_perp_squared_0) for r in r_ratios]
    )

# =========================================================
# 連結プロット：上段を下段より少し大きく
# =========================================================
fig, axes = plt.subplots(
    nrows=2, ncols=1, sharex=True, figsize=(15, 10),
    gridspec_kw={
        'hspace': 0.05,           # パネル間の縦スペース
        'height_ratios': [1.7, 1] # ← ここで比率を指定（[上段, 下段]）
    })

# -------------------------
# (1) 上段: γmax / ω0 & Firehose
# -------------------------
ax1  = axes[0]
ax1_r = ax1.twinx()

ax1.plot(r_ratios_cutoff, max_imaginary_parts_CGL,
         color='tab:red', lw=2, label=r"$\gamma_{max}/\omega_{0}$ (CGL)")
ax1.plot(r_ratios, max_imaginary_parts_iso,
         color='tab:green', lw=2, label=r"$\gamma_{max}/\omega_{0}$ (isotropic)")
ax1_r.plot(r_ratios, firehose_condition,
           color='tab:blue', lw=2, label="Discriminant value")

ax1.set_ylabel(r"$\gamma_{max}/\omega_{0}$", fontsize=22)
ax1.tick_params(axis='y', labelsize=18)
ax1.grid(True, alpha=0.3)
ax1_r.set_ylabel("Discriminant value", fontsize=22, color='tab:blue')
ax1_r.tick_params(axis='y', labelsize=18, colors='tab:blue')
ax1_r.set_ylim(-1, 1.5)

# 上段凡例（右外）
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_r.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,
           loc='center left', bbox_to_anchor=(1.20, 0.5),
           fontsize=16, frameon=False)

# -------------------------
# (2) 下段: β∥, β, ξ, B⊥²
# -------------------------
ax2 = axes[1]

beta_parallel_vals = beta_parallel_0 * r_ratios**(a)
beta_iso_vals      = beta_0 * r_ratios**(d)
xi_vals            = xi_0 * r_ratios**(b)
B_perp_vals        = B_perp_squared_0 * r_ratios**(c)

ax2.plot(r_ratios, beta_iso_vals,      lw=2, color='tab:brown',
            label=r'$\beta$ (isotropic)')
ax2.plot(r_ratios, beta_parallel_vals, lw=2, color='tab:purple',
            label=r'$\beta_{\parallel}$ (CGL)')
ax2.plot(r_ratios, xi_vals,            lw=2, color='tab:orange',
            label=r'$\xi=T_{\perp}/T_{\parallel}$')
ax2.plot(r_ratios, B_perp_vals,        lw=2, color='tab:cyan',
            label=r'$\hat{B}_{\perp}^{2}$')

ax2.set_yscale('log')
ax2.set_xlabel(r'$R/R_{0}$')
ax2.set_ylabel(r'$\beta,\;\beta_{\parallel},\;\xi,\;\hat{B}_{\perp}^{2}$')
ax2.set_ylim(1e-5, 1e2)
ax2.set_yticks([1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2])
from matplotlib.ticker import LogLocator
# ───── log‐grid 追加ここから ─────
ax2.yaxis.set_minor_locator(LogLocator(base=10.0,          # 10 のべき
                                       subs=np.arange(1, 10)*0.1,  # 1–9 の副目盛り
                                       numticks=100))
ax2.grid(True,  which='major', alpha=0.6)  # メジャー
ax2.grid(True,  which='minor', axis='y', alpha=0.3)  # マイナー
# ───── ここまで ─────
ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

# -------------------------
# 図全体レイアウト
# -------------------------
plt.subplots_adjust(right=0.8)  # 右 20 % を凡例スペースに
plt.show()

# ===============================
# 4. 各 r_ratio に対するデータセット (90×3) の作成と出力
# ===============================
# 出力するデータセットの各行は [max_imaginary_CGL, max_imaginary_isotropic, firehose_condition] となる
# ※ CGL の値は、firehose 条件が正の範囲 (r < r_ratios_cutoff) では計算済み。条件が成立しない r_ratio には NaN を補完

# 90 行のデータセットを作成
dataset = np.empty((len(r_ratios), 3))
dataset[:, 1] = max_imaginary_parts_iso          # 等方モデルの値は全 r_ratio に対してある
dataset[:, 2] = firehose_condition               # Firehose 条件

# CGL の値は、cutoff_index まで計算されているので、それ以降は NaN とする
dataset[:cutoff_index, 0] = max_imaginary_parts_CGL
dataset[cutoff_index:, 0] = np.nan

# dataset の各行に対応する r_ratio の値も表示したい場合は、別途 r_ratios を利用するか、カラムとして結合する
# 例: [r_ratio, max_imaginary_CGL, max_imaginary_isotropic, firehose_condition] の 90×4 のデータセット
# 今回は 90×3 (各モデルの値のみ) を出力する

# データセットの出力（例：標準出力に表示）
print("行: r_ratio のインデックス (1～90)")
print("列: [γₘₐₓ/ω₀ (CGL), γₘₐₓ/ω₀ (isotropic), Firehose 条件]")
print(dataset)

# -------------------------
# 追加: 下段プロット用データも含めて保存
# -------------------------
# 8 列: r, γCGL, γiso, Firehose, β∥, β, ξ, B⊥²
full_dataset = np.column_stack([
    r_ratios,
    dataset,                   # 既存の 3 列
    beta_parallel_vals,
    beta_iso_vals,
    xi_vals,
    B_perp_vals
])

# ヘッダー行を列名に合わせて作成
header = ("r_ratio, gamma_CGL, gamma_iso, firehose, "
          "beta_parallel, beta_iso, xi, B_perp_squared")

# 標準出力で確認
print("\n--- full_dataset preview ---")
print(full_dataset[:5])   # 先頭 5 行だけ表示

# CSV 保存
np.savetxt("Fig2a_dataset_full.csv", full_dataset,
           delimiter=",", header=header, comments="")
print(">> Fig2a_dataset_full.csv を保存しました")

#シナリオ2　PSPデータに基づいたBとρのスケーリングを使った断熱膨張シナリオ beta=0.01ver　完成版

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import matplotlib.ticker as mticker

# ===============================
# 1. モデル（CGL, adiabatic, WKB）の設定
# ===============================
xi_0 = 10
beta_parallel_0 = 0.01/7
B_perp_squared_0 = 0.001

a = -1.13
b = 0.2
c = 0.92
d = -2.59*(5/3)+3.32

def calculate_max_imaginary_part_CGL(r_ratio, xi_0, beta_parallel_0):
    # r_ratio によりパラメータ更新
    beta_parallel = beta_parallel_0 * r_ratio**(a)
    xi = xi_0 * r_ratio**(b)
    B_perp_squared = B_perp_squared_0 * r_ratio**(c)
    tilde_beta = beta_parallel * (3.0 / 2.0) / (1.0 + B_perp_squared + beta_parallel/2.0*(xi-1.0))

    def newton_method(omega_hat, k_hat):
        f = ((omega_hat**2 - tilde_beta * k_hat**2 * (1.0 + B_perp_squared * xi / 3.0)) *
             ((omega_hat - k_hat) * ((omega_hat + k_hat)**2 - 4.0) +
              tilde_beta * B_perp_squared * (xi - 4.0) / (3.0 * (1.0 + B_perp_squared)) *
              ((k_hat**2 + 1.0) * omega_hat + k_hat * (k_hat**2 - 3.0))) -
             (B_perp_squared * k_hat**2 * (1.0 - tilde_beta * (3.0 - xi - B_perp_squared) / (3.0 * (1.0 + B_perp_squared))) *
              ((omega_hat**3 + omega_hat**2 * k_hat - 3.0 * omega_hat + k_hat) -
               tilde_beta * (3.0 - xi) / 3.0 * ((k_hat**2 + 1.0) * omega_hat + k_hat * (k_hat**2 - 3.0)))))
        df = (2 * omega_hat * ((omega_hat - k_hat) * ((omega_hat + k_hat)**2 - 4.0) +
                             tilde_beta * B_perp_squared * (xi - 4.0) / (3.0 * (1.0 + B_perp_squared)) *
                             ((k_hat**2 + 1.0) * omega_hat + k_hat * (k_hat**2 - 3.0)))
            + (omega_hat**2 - tilde_beta * k_hat**2 * (1.0 + B_perp_squared * xi / 3.0)) *
            (((omega_hat + k_hat)**2 - 4.0) + (omega_hat - k_hat) * 2 * (omega_hat + k_hat))
            + tilde_beta * B_perp_squared * (xi - 4.0) / (3.0 * (1.0 + B_perp_squared)) * (k_hat**2 + 1.0)
            - B_perp_squared * k_hat**2 * (1.0 - tilde_beta * (3.0 - xi - B_perp_squared) / (3.0 * (1.0 + B_perp_squared))) *
            (3 * omega_hat**2 + 2 * omega_hat * k_hat - 3.0)
            - (B_perp_squared * k_hat**2 * (-tilde_beta * (3.0 - xi) / 3.0)) * (k_hat**2 + 1.0) )
        if np.abs(df) < 1e-10:
            return omega_hat
        return omega_hat - f / df

    def root_of_func(initial_guesses, k_hat):
        roots = []
        for omega_hat in initial_guesses:
            for _ in range(1000):
                omega_hat_new = newton_method(omega_hat, k_hat)
                if np.abs(omega_hat_new - omega_hat) < 1e-10:
                    if all(np.abs(omega_hat_new - root) > 1e-10 for root in roots):
                        roots.append(omega_hat_new)
                    break
                omega_hat = omega_hat_new
        return roots

    k_hat_values = np.linspace(0.95, 2.5, 1000)
    initial_guesses = [0.1+0.2j, 2+0.5j, -5+2j, -5-8j, 7-3j, 1+1j, 2+0.1j, 2-0.1j,
                       -2+2j, -2-2j, 3+3j, 3-3j, -4+0.5j, 4+0.5j, 0+3j, 0-3j,
                       1+2j, 1-2j, -1+0.5j, -1-0.5j, 0.5+1j, 0.5-1j, -0.5+0.2j,
                       -0.5-0.2j, 3.5+4j, 3.5-4j, -3.5+4j, -3.5-4j,
                       2.15+0.1j, 2.20+0.1j, 2.25+0.1j, 2.1+0.1j, 2.15+0.07j, 2.15+0.04j, 2.15+0.01j, -4-4j,
                       6+2j, 6-2j, -6+2j, -6-2j, 0.1+0.3j, 0.1-0.3j, -0.1+0.3j, -0.1-0.3j]
    omega_hats_imag = []
    for k_hat in k_hat_values:
        roots = root_of_func(initial_guesses, k_hat)
        for root in roots:
            omega_hats_imag.append(root.imag)
    return np.max(omega_hats_imag)

def calculate_firehose_condition(r_ratios, xi_0, beta_parallel_0, B_perp_squared_0):
    xi_values = xi_0 * r_ratios**(b)
    beta_parallel_values = beta_parallel_0 * r_ratios**(a)
    B_perp_squared_values = B_perp_squared_0 * r_ratios**(c)
    condition = 1 + 0.5 * beta_parallel_values * (xi_values - 1) / (1 + B_perp_squared_values)
    return condition

# ===============================
# 2. モデル（等方的: B,ρ ~ R⁻², Tは pρ^(-5/3)=const, adiabatic, WKB）の設定
# ===============================
beta_0 = 0.01
def calculate_max_imaginary_part_isotropic(args):
    r_ratio, B_perp_squared_0 = args
    beta = beta_0 * r_ratio**(d)
    B_perp_squared = B_perp_squared_0 * r_ratio
    def newton_method(omega_hat, k_hat):
        f = (omega_hat - k_hat) * (omega_hat**2 - beta * k_hat**2) * ((omega_hat + k_hat)**2 - 4) \
            - (B_perp_squared * k_hat**2) * (omega_hat**3 + k_hat * omega_hat**2 - 3 * omega_hat + k_hat)
        df = (3 * B_perp_squared * k_hat**2 + beta * k_hat**4 + 4 * beta * k_hat**2 +
              4 * k_hat * omega_hat**3 + 5 * omega_hat**4 +
              3 * omega_hat**2 * (-B_perp_squared * k_hat**2 - beta * k_hat**2 - k_hat**2 - 4) +
              2 * omega_hat * (-B_perp_squared * k_hat**3 - beta * k_hat**3 - k_hat**3 + 4 * k_hat))
        if np.abs(df) < 1e-10:
            return omega_hat
        return omega_hat - f / df
    def root_of_func(initial_guesses, k_hat):
        roots = []
        for omega_hat in initial_guesses:
            for _ in range(300):
                omega_hat_new = newton_method(omega_hat, k_hat)
                if np.abs(omega_hat_new - omega_hat) < 1e-10:
                    if all(np.abs(omega_hat_new - root) > 1e-10 for root in roots):
                        roots.append(omega_hat_new)
                    break
                omega_hat = omega_hat_new
        return roots
    k_hat_values = np.linspace(0.85, 2.5, 1000)
    initial_guesses = [0.1+0.2j, 2+0.5j, -5+2j, -5-8j, 7-3j, 1+1j, 2+0.1j, 2-0.1j,
                       -2+2j, -2-2j, 3+3j, 3-3j, -4+0.5j, 4+0.5j, 0+3j, 0-3j,
                       1+2j, 1-2j, -1+0.5j, -1-0.5j, 0.5+1j, 0.5-1j, -0.5+0.2j,
                       -0.5-0.2j, 3.5+4j, 3.5-4j, -3.5+4j, -3.5-4j,
                       2.15+0.1j, 2.20+0.1j, 2.25+0.1j, 2.1+0.1j, 2.15+0.07j, 2.15+0.04j, 2.15+0.01j, -4-4j,
                       6+2j, 6-2j, -6+2j, -6-2j, 0.1+0.3j, 0.1-0.3j, -0.1+0.3j, -0.1-0.3j]
    omega_hats_imag = []
    for k_hat in k_hat_values:
        roots = root_of_func(initial_guesses, k_hat)
        for root in roots:
            omega_hats_imag.append(root.imag)
    return np.max(omega_hats_imag)

# ===============================
# 3. 各 r_ratio での計算とプロット
# ===============================
r_ratios = np.linspace(1, 30, 90)

# (A) Firehose 条件（CGLモデル）の計算
firehose_condition = calculate_firehose_condition(r_ratios, xi_0, beta_parallel_0, B_perp_squared_0)
# Firehose 条件が 0 未満となる最初の r_ratio インデックス（条件成立しなくなるところ）
cutoff_index = np.argmax(firehose_condition < 0) if np.any(firehose_condition < 0) else len(r_ratios)
r_ratios_cutoff = r_ratios[:cutoff_index]

# (B) CGLモデルの最大虚部 γₘₐₓ/ω₀ の計算（並列処理）
with Pool() as pool:
    max_imaginary_parts_CGL = pool.starmap(
        calculate_max_imaginary_part_CGL,
        [(r, xi_0, beta_parallel_0) for r in r_ratios_cutoff]
    )

# (C) 等方モデルの最大虚部 γₘₐₓ/ω₀ の計算（並列処理）
with Pool() as pool:
    max_imaginary_parts_iso = pool.map(
        calculate_max_imaginary_part_isotropic,
        [(r, B_perp_squared_0) for r in r_ratios]
    )

# =========================================================
# 連結プロット：上段を下段より少し大きく
# =========================================================
fig, axes = plt.subplots(
    nrows=2, ncols=1, sharex=True, figsize=(15, 10),
    gridspec_kw={
        'hspace': 0.05,           # パネル間の縦スペース
        'height_ratios': [1.7, 1] # ← ここで比率を指定（[上段, 下段]）
    })

# -------------------------
# (1) 上段: γmax / ω0 & Firehose
# -------------------------
ax1  = axes[0]
ax1_r = ax1.twinx()

ax1.plot(r_ratios_cutoff, max_imaginary_parts_CGL,
         color='tab:red', lw=2, label=r"$\gamma_{max}/\omega_{0}$ (CGL)")
ax1.plot(r_ratios, max_imaginary_parts_iso,
         color='tab:green', lw=2, label=r"$\gamma_{max}/\omega_{0}$ (isotropic)")
ax1_r.plot(r_ratios, firehose_condition,
           color='tab:blue', lw=2, label="Discriminant value")

ax1.set_ylabel(r"$\gamma_{max}/\omega_{0}$", fontsize=22)
ax1.tick_params(axis='y', labelsize=18)
ax1.grid(True, alpha=0.3)
ax1_r.set_ylabel("Discriminant value", fontsize=22, color='tab:blue')
ax1_r.tick_params(axis='y', labelsize=18, colors='tab:blue')
ax1_r.set_ylim(-1, 1.5)

# 上段凡例（右外）
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_r.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,
           loc='center left', bbox_to_anchor=(1.20, 0.5),
           fontsize=16, frameon=False)

# -------------------------
# (2) 下段: β∥, β, ξ, B⊥²
# -------------------------
ax2 = axes[1]

beta_parallel_vals = beta_parallel_0 * r_ratios**(a)
beta_iso_vals      = beta_0 * r_ratios**(d)
xi_vals            = xi_0 * r_ratios**(b)
B_perp_vals        = B_perp_squared_0 * r_ratios**(c)

ax2.plot(r_ratios, beta_iso_vals,      lw=2, color='tab:brown',
            label=r'$\beta$ (isotropic)')
ax2.plot(r_ratios, beta_parallel_vals, lw=2, color='tab:purple',
            label=r'$\beta_{\parallel}$ (CGL)')
ax2.plot(r_ratios, xi_vals,            lw=2, color='tab:orange',
            label=r'$\xi=T_{\perp}/T_{\parallel}$')
ax2.plot(r_ratios, B_perp_vals,        lw=2, color='tab:cyan',
            label=r'$\hat{B}_{\perp}^{2}$')

ax2.set_yscale('log')
ax2.set_xlabel(r'$R/R_{0}$')
ax2.set_ylabel(r'$\beta,\;\beta_{\parallel},\;\xi,\;\hat{B}_{\perp}^{2}$')
ax2.set_ylim(1e-5, 1e2)
ax2.set_yticks([1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2])
from matplotlib.ticker import LogLocator
# ───── log‐grid 追加ここから ─────
ax2.yaxis.set_minor_locator(LogLocator(base=10.0,          # 10 のべき
                                       subs=np.arange(1, 10)*0.1,  # 1–9 の副目盛り
                                       numticks=100))
ax2.grid(True,  which='major', alpha=0.6)  # メジャー
ax2.grid(True,  which='minor', axis='y', alpha=0.3)  # マイナー
# ───── ここまで ─────
ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

# -------------------------
# 図全体レイアウト
# -------------------------
plt.subplots_adjust(right=0.8)  # 右 20 % を凡例スペースに
plt.show()

# ===============================
# 4. 各 r_ratio に対するデータセット (90×3) の作成と出力
# ===============================
# 出力するデータセットの各行は [max_imaginary_CGL, max_imaginary_isotropic, firehose_condition] となる
# ※ CGL の値は、firehose 条件が正の範囲 (r < r_ratios_cutoff) では計算済み。条件が成立しない r_ratio には NaN を補完

# 90 行のデータセットを作成
dataset = np.empty((len(r_ratios), 3))
dataset[:, 1] = max_imaginary_parts_iso          # 等方モデルの値は全 r_ratio に対してある
dataset[:, 2] = firehose_condition               # Firehose 条件

# CGL の値は、cutoff_index まで計算されているので、それ以降は NaN とする
dataset[:cutoff_index, 0] = max_imaginary_parts_CGL
dataset[cutoff_index:, 0] = np.nan

# dataset の各行に対応する r_ratio の値も表示したい場合は、別途 r_ratios を利用するか、カラムとして結合する
# 例: [r_ratio, max_imaginary_CGL, max_imaginary_isotropic, firehose_condition] の 90×4 のデータセット
# 今回は 90×3 (各モデルの値のみ) を出力する

# データセットの出力（例：標準出力に表示）
print("行: r_ratio のインデックス (1～90)")
print("列: [γₘₐₓ/ω₀ (CGL), γₘₐₓ/ω₀ (isotropic), Firehose 条件]")
print(dataset)

# -------------------------
# 追加: 下段プロット用データも含めて保存
# -------------------------
# 8 列: r, γCGL, γiso, Firehose, β∥, β, ξ, B⊥²
full_dataset = np.column_stack([
    r_ratios,
    dataset,                   # 既存の 3 列
    beta_parallel_vals,
    beta_iso_vals,
    xi_vals,
    B_perp_vals
])

# ヘッダー行を列名に合わせて作成
header = ("r_ratio, gamma_CGL, gamma_iso, firehose, "
          "beta_parallel, beta_iso, xi, B_perp_squared")

# 標準出力で確認
print("\n--- full_dataset preview ---")
print(full_dataset[:5])   # 先頭 5 行だけ表示

# CSV 保存
np.savetxt("Fig2b_dataset_full.csv", full_dataset,
           delimiter=",", header=header, comments="")
print(">> Fig2b_dataset_full.csv を保存しました")

#シナリオ2　PSPデータに基づいたBとρのスケーリングを使った断熱膨張シナリオ beta=0.1ver　完成版

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import matplotlib.ticker as mticker

# ===============================
# 1. モデル（CGL, adiabatic, WKB）の設定
# ===============================
xi_0 = 10
beta_parallel_0 = 0.1/7
B_perp_squared_0 = 0.001

a = -1.13
b = 0.2
c = 0.92
d = -2.59*(5/3)+3.32

def calculate_max_imaginary_part_CGL(r_ratio, xi_0, beta_parallel_0):
    # r_ratio によりパラメータ更新
    beta_parallel = beta_parallel_0 * r_ratio**(a)
    xi = xi_0 * r_ratio**(b)
    B_perp_squared = B_perp_squared_0 * r_ratio**(c)
    tilde_beta = beta_parallel * (3.0 / 2.0) / (1.0 + B_perp_squared + beta_parallel/2.0*(xi-1.0))

    def newton_method(omega_hat, k_hat):
        f = ((omega_hat**2 - tilde_beta * k_hat**2 * (1.0 + B_perp_squared * xi / 3.0)) *
             ((omega_hat - k_hat) * ((omega_hat + k_hat)**2 - 4.0) +
              tilde_beta * B_perp_squared * (xi - 4.0) / (3.0 * (1.0 + B_perp_squared)) *
              ((k_hat**2 + 1.0) * omega_hat + k_hat * (k_hat**2 - 3.0))) -
             (B_perp_squared * k_hat**2 * (1.0 - tilde_beta * (3.0 - xi - B_perp_squared) / (3.0 * (1.0 + B_perp_squared))) *
              ((omega_hat**3 + omega_hat**2 * k_hat - 3.0 * omega_hat + k_hat) -
               tilde_beta * (3.0 - xi) / 3.0 * ((k_hat**2 + 1.0) * omega_hat + k_hat * (k_hat**2 - 3.0)))))
        df = (2 * omega_hat * ((omega_hat - k_hat) * ((omega_hat + k_hat)**2 - 4.0) +
                             tilde_beta * B_perp_squared * (xi - 4.0) / (3.0 * (1.0 + B_perp_squared)) *
                             ((k_hat**2 + 1.0) * omega_hat + k_hat * (k_hat**2 - 3.0)))
            + (omega_hat**2 - tilde_beta * k_hat**2 * (1.0 + B_perp_squared * xi / 3.0)) *
            (((omega_hat + k_hat)**2 - 4.0) + (omega_hat - k_hat) * 2 * (omega_hat + k_hat))
            + tilde_beta * B_perp_squared * (xi - 4.0) / (3.0 * (1.0 + B_perp_squared)) * (k_hat**2 + 1.0)
            - B_perp_squared * k_hat**2 * (1.0 - tilde_beta * (3.0 - xi - B_perp_squared) / (3.0 * (1.0 + B_perp_squared))) *
            (3 * omega_hat**2 + 2 * omega_hat * k_hat - 3.0)
            - (B_perp_squared * k_hat**2 * (-tilde_beta * (3.0 - xi) / 3.0)) * (k_hat**2 + 1.0) )
        if np.abs(df) < 1e-10:
            return omega_hat
        return omega_hat - f / df

    def root_of_func(initial_guesses, k_hat):
        roots = []
        for omega_hat in initial_guesses:
            for _ in range(1000):
                omega_hat_new = newton_method(omega_hat, k_hat)
                if np.abs(omega_hat_new - omega_hat) < 1e-10:
                    if all(np.abs(omega_hat_new - root) > 1e-10 for root in roots):
                        roots.append(omega_hat_new)
                    break
                omega_hat = omega_hat_new
        return roots

    k_hat_values = np.linspace(0.95, 2.5, 1000)
    initial_guesses = [0.1+0.2j, 2+0.5j, -5+2j, -5-8j, 7-3j, 1+1j, 2+0.1j, 2-0.1j,
                       -2+2j, -2-2j, 3+3j, 3-3j, -4+0.5j, 4+0.5j, 0+3j, 0-3j,
                       1+2j, 1-2j, -1+0.5j, -1-0.5j, 0.5+1j, 0.5-1j, -0.5+0.2j,
                       -0.5-0.2j, 3.5+4j, 3.5-4j, -3.5+4j, -3.5-4j,
                       2.15+0.1j, 2.20+0.1j, 2.25+0.1j, 2.1+0.1j, 2.15+0.07j, 2.15+0.04j, 2.15+0.01j, -4-4j,
                       6+2j, 6-2j, -6+2j, -6-2j, 0.1+0.3j, 0.1-0.3j, -0.1+0.3j, -0.1-0.3j]
    omega_hats_imag = []
    for k_hat in k_hat_values:
        roots = root_of_func(initial_guesses, k_hat)
        for root in roots:
            omega_hats_imag.append(root.imag)
    return np.max(omega_hats_imag)

def calculate_firehose_condition(r_ratios, xi_0, beta_parallel_0, B_perp_squared_0):
    xi_values = xi_0 * r_ratios**(b)
    beta_parallel_values = beta_parallel_0 * r_ratios**(a)
    B_perp_squared_values = B_perp_squared_0 * r_ratios**(c)
    condition = 1 + 0.5 * beta_parallel_values * (xi_values - 1) / (1 + B_perp_squared_values)
    return condition

# ===============================
# 2. モデル（等方的: B,ρ ~ R⁻², Tは pρ^(-5/3)=const, adiabatic, WKB）の設定
# ===============================
beta_0 = 0.1
def calculate_max_imaginary_part_isotropic(args):
    r_ratio, B_perp_squared_0 = args
    beta = beta_0 * r_ratio**(d)
    B_perp_squared = B_perp_squared_0 * r_ratio
    def newton_method(omega_hat, k_hat):
        f = (omega_hat - k_hat) * (omega_hat**2 - beta * k_hat**2) * ((omega_hat + k_hat)**2 - 4) \
            - (B_perp_squared * k_hat**2) * (omega_hat**3 + k_hat * omega_hat**2 - 3 * omega_hat + k_hat)
        df = (3 * B_perp_squared * k_hat**2 + beta * k_hat**4 + 4 * beta * k_hat**2 +
              4 * k_hat * omega_hat**3 + 5 * omega_hat**4 +
              3 * omega_hat**2 * (-B_perp_squared * k_hat**2 - beta * k_hat**2 - k_hat**2 - 4) +
              2 * omega_hat * (-B_perp_squared * k_hat**3 - beta * k_hat**3 - k_hat**3 + 4 * k_hat))
        if np.abs(df) < 1e-10:
            return omega_hat
        return omega_hat - f / df
    def root_of_func(initial_guesses, k_hat):
        roots = []
        for omega_hat in initial_guesses:
            for _ in range(300):
                omega_hat_new = newton_method(omega_hat, k_hat)
                if np.abs(omega_hat_new - omega_hat) < 1e-10:
                    if all(np.abs(omega_hat_new - root) > 1e-10 for root in roots):
                        roots.append(omega_hat_new)
                    break
                omega_hat = omega_hat_new
        return roots
    k_hat_values = np.linspace(0.85, 2.5, 1000)
    initial_guesses = [0.1+0.2j, 2+0.5j, -5+2j, -5-8j, 7-3j, 1+1j, 2+0.1j, 2-0.1j,
                       -2+2j, -2-2j, 3+3j, 3-3j, -4+0.5j, 4+0.5j, 0+3j, 0-3j,
                       1+2j, 1-2j, -1+0.5j, -1-0.5j, 0.5+1j, 0.5-1j, -0.5+0.2j,
                       -0.5-0.2j, 3.5+4j, 3.5-4j, -3.5+4j, -3.5-4j,
                       2.15+0.1j, 2.20+0.1j, 2.25+0.1j, 2.1+0.1j, 2.15+0.07j, 2.15+0.04j, 2.15+0.01j, -4-4j,
                       6+2j, 6-2j, -6+2j, -6-2j, 0.1+0.3j, 0.1-0.3j, -0.1+0.3j, -0.1-0.3j]
    omega_hats_imag = []
    for k_hat in k_hat_values:
        roots = root_of_func(initial_guesses, k_hat)
        for root in roots:
            omega_hats_imag.append(root.imag)
    return np.max(omega_hats_imag)

# ===============================
# 3. 各 r_ratio での計算とプロット
# ===============================
r_ratios = np.linspace(1, 30, 90)

# (A) Firehose 条件（CGLモデル）の計算
firehose_condition = calculate_firehose_condition(r_ratios, xi_0, beta_parallel_0, B_perp_squared_0)
# Firehose 条件が 0 未満となる最初の r_ratio インデックス（条件成立しなくなるところ）
cutoff_index = np.argmax(firehose_condition < 0) if np.any(firehose_condition < 0) else len(r_ratios)
r_ratios_cutoff = r_ratios[:cutoff_index]

# (B) CGLモデルの最大虚部 γₘₐₓ/ω₀ の計算（並列処理）
with Pool() as pool:
    max_imaginary_parts_CGL = pool.starmap(
        calculate_max_imaginary_part_CGL,
        [(r, xi_0, beta_parallel_0) for r in r_ratios_cutoff]
    )

# (C) 等方モデルの最大虚部 γₘₐₓ/ω₀ の計算（並列処理）
with Pool() as pool:
    max_imaginary_parts_iso = pool.map(
        calculate_max_imaginary_part_isotropic,
        [(r, B_perp_squared_0) for r in r_ratios]
    )

# =========================================================
# 連結プロット：上段を下段より少し大きく
# =========================================================
fig, axes = plt.subplots(
    nrows=2, ncols=1, sharex=True, figsize=(15, 10),
    gridspec_kw={
        'hspace': 0.05,           # パネル間の縦スペース
        'height_ratios': [1.7, 1] # ← ここで比率を指定（[上段, 下段]）
    })

# -------------------------
# (1) 上段: γmax / ω0 & Firehose
# -------------------------
ax1  = axes[0]
ax1_r = ax1.twinx()

ax1.plot(r_ratios_cutoff, max_imaginary_parts_CGL,
         color='tab:red', lw=2, label=r"$\gamma_{max}/\omega_{0}$ (CGL)")
ax1.plot(r_ratios, max_imaginary_parts_iso,
         color='tab:green', lw=2, label=r"$\gamma_{max}/\omega_{0}$ (isotropic)")
ax1_r.plot(r_ratios, firehose_condition,
           color='tab:blue', lw=2, label="Discriminant value")

ax1.set_ylabel(r"$\gamma_{max}/\omega_{0}$", fontsize=22)
ax1.tick_params(axis='y', labelsize=18)
ax1.grid(True, alpha=0.3)
ax1_r.set_ylabel("Discriminant value", fontsize=22, color='tab:blue')
ax1_r.tick_params(axis='y', labelsize=18, colors='tab:blue')
ax1_r.set_ylim(-1, 1.5)

# 上段凡例（右外）
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_r.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,
           loc='center left', bbox_to_anchor=(1.20, 0.5),
           fontsize=16, frameon=False)

# -------------------------
# (2) 下段: β∥, β, ξ, B⊥²
# -------------------------
ax2 = axes[1]

beta_parallel_vals = beta_parallel_0 * r_ratios**(a)
beta_iso_vals      = beta_0 * r_ratios**(d)
xi_vals            = xi_0 * r_ratios**(b)
B_perp_vals        = B_perp_squared_0 * r_ratios**(c)

ax2.plot(r_ratios, beta_iso_vals,      lw=2, color='tab:brown',
            label=r'$\beta$ (isotropic)')
ax2.plot(r_ratios, beta_parallel_vals, lw=2, color='tab:purple',
            label=r'$\beta_{\parallel}$ (CGL)')
ax2.plot(r_ratios, xi_vals,            lw=2, color='tab:orange',
            label=r'$\xi=T_{\perp}/T_{\parallel}$')
ax2.plot(r_ratios, B_perp_vals,        lw=2, color='tab:cyan',
            label=r'$\hat{B}_{\perp}^{2}$')

ax2.set_yscale('log')
ax2.set_xlabel(r'$R/R_{0}$')
ax2.set_ylabel(r'$\beta,\;\beta_{\parallel},\;\xi,\;\hat{B}_{\perp}^{2}$')
ax2.set_ylim(1e-5, 1e2)
ax2.set_yticks([1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2])
from matplotlib.ticker import LogLocator
# ───── log‐grid 追加ここから ─────
ax2.yaxis.set_minor_locator(LogLocator(base=10.0,          # 10 のべき
                                       subs=np.arange(1, 10)*0.1,  # 1–9 の副目盛り
                                       numticks=100))
ax2.grid(True,  which='major', alpha=0.6)  # メジャー
ax2.grid(True,  which='minor', axis='y', alpha=0.3)  # マイナー
# ───── ここまで ─────
ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

# -------------------------
# 図全体レイアウト
# -------------------------
plt.subplots_adjust(right=0.8)  # 右 20 % を凡例スペースに
plt.show()

# ===============================
# 4. 各 r_ratio に対するデータセット (90×3) の作成と出力
# ===============================
# 出力するデータセットの各行は [max_imaginary_CGL, max_imaginary_isotropic, firehose_condition] となる
# ※ CGL の値は、firehose 条件が正の範囲 (r < r_ratios_cutoff) では計算済み。条件が成立しない r_ratio には NaN を補完

# 90 行のデータセットを作成
dataset = np.empty((len(r_ratios), 3))
dataset[:, 1] = max_imaginary_parts_iso          # 等方モデルの値は全 r_ratio に対してある
dataset[:, 2] = firehose_condition               # Firehose 条件

# CGL の値は、cutoff_index まで計算されているので、それ以降は NaN とする
dataset[:cutoff_index, 0] = max_imaginary_parts_CGL
dataset[cutoff_index:, 0] = np.nan

# dataset の各行に対応する r_ratio の値も表示したい場合は、別途 r_ratios を利用するか、カラムとして結合する
# 例: [r_ratio, max_imaginary_CGL, max_imaginary_isotropic, firehose_condition] の 90×4 のデータセット
# 今回は 90×3 (各モデルの値のみ) を出力する

# データセットの出力（例：標準出力に表示）
print("行: r_ratio のインデックス (1～90)")
print("列: [γₘₐₓ/ω₀ (CGL), γₘₐₓ/ω₀ (isotropic), Firehose 条件]")
print(dataset)

# -------------------------
# 追加: 下段プロット用データも含めて保存
# -------------------------
# 8 列: r, γCGL, γiso, Firehose, β∥, β, ξ, B⊥²
full_dataset = np.column_stack([
    r_ratios,
    dataset,                   # 既存の 3 列
    beta_parallel_vals,
    beta_iso_vals,
    xi_vals,
    B_perp_vals
])

# ヘッダー行を列名に合わせて作成
header = ("r_ratio, gamma_CGL, gamma_iso, firehose, "
          "beta_parallel, beta_iso, xi, B_perp_squared")

# 標準出力で確認
print("\n--- full_dataset preview ---")
print(full_dataset[:5])   # 先頭 5 行だけ表示

# CSV 保存
np.savetxt("Fig2c_dataset_full.csv", full_dataset,
           delimiter=",", header=header, comments="")
print(">> Fig2c_dataset_full.csv を保存しました")

# シナリオ３　平行方向等温で垂直方向断熱、磁場と密度はPSPスケーリング　beta=0.001　本当の完成版

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import matplotlib.ticker as mticker

# ===== ここから追加／置き換え ===================================
plt.rcParams.update({
    'font.size': 20,        # デフォルトのフォントサイズ
    'axes.titlesize': 20,   # タイトル
    'axes.labelsize': 20,   # 軸ラベル
    'xtick.labelsize': 20,  # x 目盛ラベル
    'ytick.labelsize': 20,  # y 目盛ラベル
    'legend.fontsize': 20   # 凡例
})
# ================================================================

flg = 18

a = 0.73
b = -1.66
c = 0.92

# 固定定数
B_perp_squared_0 = 0.001  # 初期 B_perp^2 の値

# CGLモデル用の初期値
beta_parallel_0 = 0.001/7  # 初期 beta_parallel_0
xi_0 = 10             # 初期 xi_0

# isotropicモデル用の定数（CGLとは別に使う）
n0 = 1e8              # 単位: cm^-3 (例)
k_b = 1.38e-16        # erg/K (例)
B0 = 50               # 初期磁場（任意単位）
#beta_0 = 0.001        # isotropic用基準 beta_0

# ------------------------------
# CGLモデルのγmax/ω0の計算関数
def calculate_max_imaginary_part_CGL(r_ratio, xi_0, beta_parallel_0):
    beta_parallel = beta_parallel_0 * r_ratio**(a)
    xi = xi_0 * r_ratio**(b)
    B_perp_squared = B_perp_squared_0 * r_ratio**(c)
    tilde_beta = beta_parallel * (3.0/2.0) / (1.0 + B_perp_squared + beta_parallel/2.0*(xi - 1.0))

    def newton_method(omega_hat, k_hat):
        f = ((omega_hat**2 - tilde_beta * k_hat**2 * (1.0 + B_perp_squared * xi / 3.0)) *
             ((omega_hat - k_hat) * ((omega_hat + k_hat)**2 - 4.0) +
              tilde_beta * B_perp_squared * (xi - 4.0) / (3.0*(1.0 + B_perp_squared)) *
              ((k_hat**2+1.0)*omega_hat + k_hat*(k_hat**2-3.0))) -
             (B_perp_squared * k_hat**2 * (1.0 - tilde_beta*(3.0 - xi - B_perp_squared) / (3.0*(1.0+B_perp_squared))) *
              ((omega_hat**3+omega_hat**2*k_hat-3.0*omega_hat+k_hat) - tilde_beta*(3.0 - xi)/3.0 *
               ((k_hat**2+1.0)*omega_hat + k_hat*(k_hat**2-3.0)))))
        df = (2 * omega_hat * ((omega_hat - k_hat)*((omega_hat+k_hat)**2 - 4.0) +
                tilde_beta*B_perp_squared*(xi-4.0)/(3.0*(1.0+B_perp_squared))*((k_hat**2+1.0)*omega_hat+k_hat*(k_hat**2-3.0))) +
              (omega_hat**2 - tilde_beta*k_hat**2*(1.0+B_perp_squared*xi/3.0))*
              (((omega_hat+k_hat)**2-4.0)+(omega_hat-k_hat)*2*(omega_hat+k_hat)) +
              tilde_beta*B_perp_squared*(xi-4.0)/(3.0*(1.0+B_perp_squared))*(k_hat**2+1.0) -
              B_perp_squared*k_hat**2*(1.0-tilde_beta*(3.0-xi-B_perp_squared)/(3.0*(1.0+B_perp_squared)))*
              (3*omega_hat**2+2*omega_hat*k_hat-3.0) -
              (B_perp_squared*k_hat**2*(-tilde_beta*(3.0-xi)/3.0))*(k_hat**2+1.0) )
        if np.abs(df) < 1e-10:
            return omega_hat
        return omega_hat - f/df

    def root_of_func(initial_guesses, k_hat):
        roots = []
        for omega_hat in initial_guesses:
            for _ in range(300):
                omega_hat_new = newton_method(omega_hat, k_hat)
                if np.abs(omega_hat_new - omega_hat) < 1e-10:
                    if all(np.abs(omega_hat_new - root) > 1e-10 for root in roots):
                        roots.append(omega_hat_new)
                    break
                omega_hat = omega_hat_new
        return roots

    k_hat_values = np.linspace(0.85, 2.5, 1000)
    initial_guesses = [0.1+0.2j, 2+0.5j, -5+2j, -5-8j, 7-3j, 1+1j, 2+0.1j, 2-0.1j,
                       -2+2j, -2-2j, 3+3j, 3-3j, -4+0.5j, 4+0.5j, 0+3j, 0-3j,
                       1+2j, 1-2j, -1+0.5j, -1-0.5j, 0.5+1j, 0.5-1j, -0.5+0.2j,
                       -0.5-0.2j, 3.5+4j, 3.5-4j, -3.5+4j, -3.5-4j,
                       2.15+0.1j, 2.20+0.1j, 2.25+0.1j, 2.1+0.1j, 2.15+0.07j, 2.15+0.04j, 2.15+0.01j, -4-4j,
                       6+2j, 6-2j, -6+2j, -6-2j, 0.1+0.3j, 0.1-0.3j, -0.1+0.3j, -0.1-0.3j]
    omega_hats_imag = []
    for k_hat in k_hat_values:
        roots = root_of_func(initial_guesses, k_hat)
        for root in roots:
            omega_hats_imag.append(root.imag)
    return np.max(omega_hats_imag)

# ------------------------------
# isotropicモデルのγmax/ω0の計算関数
def calculate_max_imaginary_part_isotropic(args):
    r_ratio, B_perp_squared_0 = args
    # 新たな β の計算（density, B, T のスケーリングに基づく）
    n = n0 * r_ratio**(-2.59)
    B = B0 * r_ratio**(-1.66)
    T = 1.03e6 * (1 + 20 * r_ratio**(-1.66)) / 3
    beta = n * k_b * T * 8 * np.pi / (B**2)

    B_perp_squared = B_perp_squared_0 * r_ratio**(c)

    def newton_method(omega_hat, k_hat):
        f = (omega_hat - k_hat) * (omega_hat**2 - beta*k_hat**2) * ((omega_hat+k_hat)**2 - 4) \
            - (B_perp_squared * k_hat**2) * (omega_hat**3 + k_hat*omega_hat**2 - 3*omega_hat + k_hat)
        df = (3 * B_perp_squared * k_hat**2 + beta*k_hat**4 + 4*beta*k_hat**2 +
              4*k_hat*omega_hat**3 + 5*omega_hat**4 +
              3*omega_hat**2 * (-B_perp_squared*k_hat**2 - beta*k_hat**2 - k_hat**2 - 4) +
              2*omega_hat * (-B_perp_squared*k_hat**3 - beta*k_hat**3 - k_hat**3 + 4*k_hat))
        if np.abs(df) < 1e-10:
            return omega_hat
        return omega_hat - f/df

    def root_of_func(initial_guesses, k_hat):
        roots = []
        for omega_hat in initial_guesses:
            for _ in range(300):
                omega_hat_new = newton_method(omega_hat, k_hat)
                if np.abs(omega_hat_new - omega_hat) < 1e-10:
                    if all(np.abs(omega_hat_new - root) > 1e-10 for root in roots):
                        roots.append(omega_hat_new)
                    break
                omega_hat = omega_hat_new
        return roots

    k_hat_values = np.linspace(0.85, 2.5, 1000)
    initial_guesses = [0.1+0.2j, 2+0.5j, -5+2j, -5-8j, 7-3j, 1+1j, 2+0.1j, 2-0.1j,
                       -2+2j, -2-2j, 3+3j, 3-3j, -4+0.5j, 4+0.5j, 0+3j, 0-3j,
                       1+2j, 1-2j, -1+0.5j, -1-0.5j, 0.5+1j, 0.5-1j, -0.5+0.2j,
                       -0.5-0.2j, 3.5+4j, 3.5-4j, -3.5+4j, -3.5-4j,
                       2.15+0.1j, 2.20+0.1j, 2.25+0.1j, 2.1+0.1j, 2.15+0.07j, 2.15+0.04j, 2.15+0.01j, -4-4j,
                       6+2j, 6-2j, -6+2j, -6-2j, 0.1+0.3j, 0.1-0.3j, -0.1+0.3j, -0.1-0.3j]
    omega_hats_imag = []
    for k_hat in k_hat_values:
        roots = root_of_func(initial_guesses, k_hat)
        for root in roots:
            omega_hats_imag.append(root.imag)
    return np.max(omega_hats_imag)

# ------------------------------
# Firehose条件（CGLモデル用）の計算関数
def calculate_firehose_condition(r_ratios, xi_0, beta_parallel_0, B_perp_squared_0):
    # xi, beta_parallel, B_perp_squared のスケーリング（CGLモデル）
    xi_values = xi_0 * r_ratios**(b)
    beta_parallel_values = beta_parallel_0 * r_ratios**(a)
    B_perp_squared_values = B_perp_squared_0 * r_ratios**(c)
    condition = 1 + 0.5 * beta_parallel_values * (xi_values - 1) / (1 + B_perp_squared_values)
    return condition

# ------------------------------
# r_ratio の範囲
r_ratios = np.linspace(1, 30, 90)

# (A) CGLモデルのγmax/ω0の計算（Firehose条件が正となる領域のみ）
firehose_condition = calculate_firehose_condition(r_ratios, xi_0, beta_parallel_0, B_perp_squared_0)
cutoff_index = np.argmax(firehose_condition < 0) if np.any(firehose_condition < 0) else len(r_ratios)
r_ratios_cutoff = r_ratios[:cutoff_index]

with Pool() as pool:
    max_imaginary_parts_CGL = pool.starmap(
        calculate_max_imaginary_part_CGL,
        [(r, xi_0, beta_parallel_0) for r in r_ratios_cutoff]
    )
    max_imaginary_parts_CGL = [result for result in max_imaginary_parts_CGL]

# (B) isotropicモデルのγmax/ω0の計算（全r_ratio）
with Pool() as pool:
    max_imaginary_parts_iso = pool.map(
        calculate_max_imaginary_part_isotropic,
        [(r, B_perp_squared_0) for r in r_ratios]
    )
    max_imaginary_parts_iso = [result for result in max_imaginary_parts_iso]

# =========================================================
# 連結プロット：上段を下段より少し大きく
# =========================================================
fig, axes = plt.subplots(
    nrows=2, ncols=1, sharex=True, figsize=(15, 10),
    gridspec_kw={
        'hspace': 0.05,           # パネル間の縦スペース
        'height_ratios': [1.7, 1] # ← ここで比率を指定（[上段, 下段]）
    })

# -------------------------
# (1) 上段: γmax / ω0 & Firehose
# -------------------------
ax1  = axes[0]
ax1_r = ax1.twinx()

ax1.plot(r_ratios_cutoff, max_imaginary_parts_CGL,
         color='tab:red', lw=2, label=r"$\gamma_{max}/\omega_{0}$ (CGL)")
ax1.plot(r_ratios, max_imaginary_parts_iso,
         color='tab:green', lw=2, label=r"$\gamma_{max}/\omega_{0}$ (isotropic)")
ax1_r.plot(r_ratios, firehose_condition,
           color='tab:blue', lw=2, label="Discriminant value")

ax1.set_ylabel(r"$\gamma_{max}/\omega_{0}$")
ax1.tick_params(axis='y')
ax1.grid(True, alpha=0.3)
ax1_r.set_ylabel("Discriminant value", color='tab:blue')
ax1_r.tick_params(axis='y',colors='tab:blue')
ax1_r.set_ylim(-1, 1.5)

# 上段凡例（右外）
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_r.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,
           loc='center left', bbox_to_anchor=(1.20, 0.5),
           frameon=False)

# -------------------------
# (2) 下段: β∥, β, ξ, B⊥²
# -------------------------
ax2 = axes[1]

beta_parallel_vals = beta_parallel_0 * r_ratios**(a)
beta_iso_vals      = beta_0 * r_ratios**(d)
xi_vals            = xi_0 * r_ratios**(b)
B_perp_vals        = B_perp_squared_0 * r_ratios**(c)

ax2.plot(r_ratios, beta_iso_vals,      lw=2, color='tab:brown',
            label=r'$\beta$ (isotropic)')
ax2.plot(r_ratios, beta_parallel_vals, lw=2, color='tab:purple',
            label=r'$\beta_{\parallel}$ (CGL)')
ax2.plot(r_ratios, xi_vals,            lw=2, color='tab:orange',
            label=r'$\xi=T_{\perp}/T_{\parallel}$')
ax2.plot(r_ratios, B_perp_vals,        lw=2, color='tab:cyan',
            label=r'$\hat{B}_{\perp}^{2}$')

ax2.set_yscale('log')
ax2.set_xlabel(r'$R/R_{0}$')
ax2.set_ylabel(r'$\beta,\;\beta_{\parallel},\;\xi,\;\hat{B}_{\perp}^{2}$')
ax2.set_ylim(1e-5, 1e2)
ax2.set_yticks([1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2])
from matplotlib.ticker import LogLocator
# ───── log‐grid 追加ここから ─────
ax2.yaxis.set_minor_locator(LogLocator(base=10.0,          # 10 のべき
                                       subs=np.arange(1, 10)*0.1,  # 1–9 の副目盛り
                                       numticks=100))
ax2.grid(True,  which='major', alpha=0.6)  # メジャー
ax2.grid(True,  which='minor', axis='y', alpha=0.3)  # マイナー
# ───── ここまで ─────
ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

# -------------------------
# 図全体レイアウト
# -------------------------
plt.subplots_adjust(right=0.8)  # 右 20 % を凡例スペースに
plt.show()

# ===============================
# 4. 各 r_ratio に対するデータセット (90×3) の作成と出力
# ===============================
# 出力するデータセットの各行は [max_imaginary_CGL, max_imaginary_isotropic, firehose_condition] となる
# ※ CGL の値は、firehose 条件が正の範囲 (r < r_ratios_cutoff) では計算済み。条件が成立しない r_ratio には NaN を補完

# 90 行のデータセットを作成
dataset = np.empty((len(r_ratios), 3))
dataset[:, 1] = max_imaginary_parts_iso          # 等方モデルの値は全 r_ratio に対してある
dataset[:, 2] = firehose_condition               # Firehose 条件

# CGL の値は、cutoff_index まで計算されているので、それ以降は NaN とする
dataset[:cutoff_index, 0] = max_imaginary_parts_CGL
dataset[cutoff_index:, 0] = np.nan

# dataset の各行に対応する r_ratio の値も表示したい場合は、別途 r_ratios を利用するか、カラムとして結合する
# 例: [r_ratio, max_imaginary_CGL, max_imaginary_isotropic, firehose_condition] の 90×4 のデータセット
# 今回は 90×3 (各モデルの値のみ) を出力する

# データセットの出力（例：標準出力に表示）
print("行: r_ratio のインデックス (1～90)")
print("列: [γₘₐₓ/ω₀ (CGL), γₘₐₓ/ω₀ (isotropic), Firehose 条件]")
print(dataset)

# -------------------------
# 追加: 下段プロット用データも含めて保存
# -------------------------
# 8 列: r, γCGL, γiso, Firehose, β∥, β, ξ, B⊥²
full_dataset = np.column_stack([
    r_ratios,
    dataset,                   # 既存の 3 列
    beta_parallel_vals,
    beta_iso_vals,
    xi_vals,
    B_perp_vals
])

# ヘッダー行を列名に合わせて作成
header = ("r_ratio, gamma_CGL, gamma_iso, firehose, "
          "beta_parallel, beta_iso, xi, B_perp_squared")

# 標準出力で確認
print("\n--- full_dataset preview ---")
print(full_dataset[:5])   # 先頭 5 行だけ表示

# CSV 保存
np.savetxt("Fig3a_dataset_full.csv", full_dataset,
           delimiter=",", header=header, comments="")
print(">> Fig3a_dataset_full.csv を保存しました")

# シナリオ３　平行方向等温で垂直方向断熱、磁場と密度はPSPスケーリング　beta=0.01　本当の完成版

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import matplotlib.ticker as mticker

# ===== ここから追加／置き換え ===================================
plt.rcParams.update({
    'font.size': 20,        # デフォルトのフォントサイズ
    'axes.titlesize': 20,   # タイトル
    'axes.labelsize': 20,   # 軸ラベル
    'xtick.labelsize': 20,  # x 目盛ラベル
    'ytick.labelsize': 20,  # y 目盛ラベル
    'legend.fontsize': 20   # 凡例
})
# ================================================================

flg = 18

a = 0.73
b = -1.66
c = 0.92

# 固定定数
B_perp_squared_0 = 0.001  # 初期 B_perp^2 の値

# CGLモデル用の初期値
beta_parallel_0 = 0.01/7  # 初期 beta_parallel_0
xi_0 = 10             # 初期 xi_0

# isotropicモデル用の定数（CGLとは別に使う）
n0 = 5e8              # 単位: cm^-3 (例)
k_b = 1.38e-16        # erg/K (例)
B0 = 35               # 初期磁場（任意単位）
#beta_0 = 0.001        # isotropic用基準 beta_0

# ------------------------------
# CGLモデルのγmax/ω0の計算関数
def calculate_max_imaginary_part_CGL(r_ratio, xi_0, beta_parallel_0):
    beta_parallel = beta_parallel_0 * r_ratio**(a)
    xi = xi_0 * r_ratio**(b)
    B_perp_squared = B_perp_squared_0 * r_ratio**(c)
    tilde_beta = beta_parallel * (3.0/2.0) / (1.0 + B_perp_squared + beta_parallel/2.0*(xi - 1.0))

    def newton_method(omega_hat, k_hat):
        f = ((omega_hat**2 - tilde_beta * k_hat**2 * (1.0 + B_perp_squared * xi / 3.0)) *
             ((omega_hat - k_hat) * ((omega_hat + k_hat)**2 - 4.0) +
              tilde_beta * B_perp_squared * (xi - 4.0) / (3.0*(1.0 + B_perp_squared)) *
              ((k_hat**2+1.0)*omega_hat + k_hat*(k_hat**2-3.0))) -
             (B_perp_squared * k_hat**2 * (1.0 - tilde_beta*(3.0 - xi - B_perp_squared) / (3.0*(1.0+B_perp_squared))) *
              ((omega_hat**3+omega_hat**2*k_hat-3.0*omega_hat+k_hat) - tilde_beta*(3.0 - xi)/3.0 *
               ((k_hat**2+1.0)*omega_hat + k_hat*(k_hat**2-3.0)))))
        df = (2 * omega_hat * ((omega_hat - k_hat)*((omega_hat+k_hat)**2 - 4.0) +
                tilde_beta*B_perp_squared*(xi-4.0)/(3.0*(1.0+B_perp_squared))*((k_hat**2+1.0)*omega_hat+k_hat*(k_hat**2-3.0))) +
              (omega_hat**2 - tilde_beta*k_hat**2*(1.0+B_perp_squared*xi/3.0))*
              (((omega_hat+k_hat)**2-4.0)+(omega_hat-k_hat)*2*(omega_hat+k_hat)) +
              tilde_beta*B_perp_squared*(xi-4.0)/(3.0*(1.0+B_perp_squared))*(k_hat**2+1.0) -
              B_perp_squared*k_hat**2*(1.0-tilde_beta*(3.0-xi-B_perp_squared)/(3.0*(1.0+B_perp_squared)))*
              (3*omega_hat**2+2*omega_hat*k_hat-3.0) -
              (B_perp_squared*k_hat**2*(-tilde_beta*(3.0-xi)/3.0))*(k_hat**2+1.0) )
        if np.abs(df) < 1e-10:
            return omega_hat
        return omega_hat - f/df

    def root_of_func(initial_guesses, k_hat):
        roots = []
        for omega_hat in initial_guesses:
            for _ in range(300):
                omega_hat_new = newton_method(omega_hat, k_hat)
                if np.abs(omega_hat_new - omega_hat) < 1e-10:
                    if all(np.abs(omega_hat_new - root) > 1e-10 for root in roots):
                        roots.append(omega_hat_new)
                    break
                omega_hat = omega_hat_new
        return roots

    k_hat_values = np.linspace(0.85, 2.5, 1000)
    initial_guesses = [0.1+0.2j, 2+0.5j, -5+2j, -5-8j, 7-3j, 1+1j, 2+0.1j, 2-0.1j,
                       -2+2j, -2-2j, 3+3j, 3-3j, -4+0.5j, 4+0.5j, 0+3j, 0-3j,
                       1+2j, 1-2j, -1+0.5j, -1-0.5j, 0.5+1j, 0.5-1j, -0.5+0.2j,
                       -0.5-0.2j, 3.5+4j, 3.5-4j, -3.5+4j, -3.5-4j,
                       2.15+0.1j, 2.20+0.1j, 2.25+0.1j, 2.1+0.1j, 2.15+0.07j, 2.15+0.04j, 2.15+0.01j, -4-4j,
                       6+2j, 6-2j, -6+2j, -6-2j, 0.1+0.3j, 0.1-0.3j, -0.1+0.3j, -0.1-0.3j]
    omega_hats_imag = []
    for k_hat in k_hat_values:
        roots = root_of_func(initial_guesses, k_hat)
        for root in roots:
            omega_hats_imag.append(root.imag)
    return np.max(omega_hats_imag)

# ------------------------------
# isotropicモデルのγmax/ω0の計算関数
def calculate_max_imaginary_part_isotropic(args):
    r_ratio, B_perp_squared_0 = args
    # 新たな β の計算（density, B, T のスケーリングに基づく）
    n = n0 * r_ratio**(-2.59)
    B = B0 * r_ratio**(-1.66)
    T = 1.03e6 * (1 + 20 * r_ratio**(-1.66)) / 3
    beta = n * k_b * T * 8 * np.pi / (B**2)

    B_perp_squared = B_perp_squared_0 * r_ratio**(c)

    def newton_method(omega_hat, k_hat):
        f = (omega_hat - k_hat) * (omega_hat**2 - beta*k_hat**2) * ((omega_hat+k_hat)**2 - 4) \
            - (B_perp_squared * k_hat**2) * (omega_hat**3 + k_hat*omega_hat**2 - 3*omega_hat + k_hat)
        df = (3 * B_perp_squared * k_hat**2 + beta*k_hat**4 + 4*beta*k_hat**2 +
              4*k_hat*omega_hat**3 + 5*omega_hat**4 +
              3*omega_hat**2 * (-B_perp_squared*k_hat**2 - beta*k_hat**2 - k_hat**2 - 4) +
              2*omega_hat * (-B_perp_squared*k_hat**3 - beta*k_hat**3 - k_hat**3 + 4*k_hat))
        if np.abs(df) < 1e-10:
            return omega_hat
        return omega_hat - f/df

    def root_of_func(initial_guesses, k_hat):
        roots = []
        for omega_hat in initial_guesses:
            for _ in range(300):
                omega_hat_new = newton_method(omega_hat, k_hat)
                if np.abs(omega_hat_new - omega_hat) < 1e-10:
                    if all(np.abs(omega_hat_new - root) > 1e-10 for root in roots):
                        roots.append(omega_hat_new)
                    break
                omega_hat = omega_hat_new
        return roots

    k_hat_values = np.linspace(0.85, 2.5, 1000)
    initial_guesses = [0.1+0.2j, 2+0.5j, -5+2j, -5-8j, 7-3j, 1+1j, 2+0.1j, 2-0.1j,
                       -2+2j, -2-2j, 3+3j, 3-3j, -4+0.5j, 4+0.5j, 0+3j, 0-3j,
                       1+2j, 1-2j, -1+0.5j, -1-0.5j, 0.5+1j, 0.5-1j, -0.5+0.2j,
                       -0.5-0.2j, 3.5+4j, 3.5-4j, -3.5+4j, -3.5-4j,
                       2.15+0.1j, 2.20+0.1j, 2.25+0.1j, 2.1+0.1j, 2.15+0.07j, 2.15+0.04j, 2.15+0.01j, -4-4j,
                       6+2j, 6-2j, -6+2j, -6-2j, 0.1+0.3j, 0.1-0.3j, -0.1+0.3j, -0.1-0.3j]
    omega_hats_imag = []
    for k_hat in k_hat_values:
        roots = root_of_func(initial_guesses, k_hat)
        for root in roots:
            omega_hats_imag.append(root.imag)
    return np.max(omega_hats_imag)

# ------------------------------
# Firehose条件（CGLモデル用）の計算関数
def calculate_firehose_condition(r_ratios, xi_0, beta_parallel_0, B_perp_squared_0):
    # xi, beta_parallel, B_perp_squared のスケーリング（CGLモデル）
    xi_values = xi_0 * r_ratios**(b)
    beta_parallel_values = beta_parallel_0 * r_ratios**(a)
    B_perp_squared_values = B_perp_squared_0 * r_ratios**(c)
    condition = 1 + 0.5 * beta_parallel_values * (xi_values - 1) / (1 + B_perp_squared_values)
    return condition

# ------------------------------
# r_ratio の範囲
r_ratios = np.linspace(1, 30, 90)

# (A) CGLモデルのγmax/ω0の計算（Firehose条件が正となる領域のみ）
firehose_condition = calculate_firehose_condition(r_ratios, xi_0, beta_parallel_0, B_perp_squared_0)
cutoff_index = np.argmax(firehose_condition < 0) if np.any(firehose_condition < 0) else len(r_ratios)
r_ratios_cutoff = r_ratios[:cutoff_index]

with Pool() as pool:
    max_imaginary_parts_CGL = pool.starmap(
        calculate_max_imaginary_part_CGL,
        [(r, xi_0, beta_parallel_0) for r in r_ratios_cutoff]
    )
    max_imaginary_parts_CGL = [result for result in max_imaginary_parts_CGL]

# (B) isotropicモデルのγmax/ω0の計算（全r_ratio）
with Pool() as pool:
    max_imaginary_parts_iso = pool.map(
        calculate_max_imaginary_part_isotropic,
        [(r, B_perp_squared_0) for r in r_ratios]
    )
    max_imaginary_parts_iso = [result for result in max_imaginary_parts_iso]

# =========================================================
# 連結プロット：上段を下段より少し大きく
# =========================================================
fig, axes = plt.subplots(
    nrows=2, ncols=1, sharex=True, figsize=(15, 10),
    gridspec_kw={
        'hspace': 0.05,           # パネル間の縦スペース
        'height_ratios': [1.7, 1] # ← ここで比率を指定（[上段, 下段]）
    })

# -------------------------
# (1) 上段: γmax / ω0 & Firehose
# -------------------------
ax1  = axes[0]
ax1_r = ax1.twinx()

ax1.plot(r_ratios_cutoff, max_imaginary_parts_CGL,
         color='tab:red', lw=2, label=r"$\gamma_{max}/\omega_{0}$ (CGL)")
ax1.plot(r_ratios, max_imaginary_parts_iso,
         color='tab:green', lw=2, label=r"$\gamma_{max}/\omega_{0}$ (isotropic)")
ax1_r.plot(r_ratios, firehose_condition,
           color='tab:blue', lw=2, label="Discriminant value")

ax1.set_ylabel(r"$\gamma_{max}/\omega_{0}$")
ax1.tick_params(axis='y')
ax1.grid(True, alpha=0.3)
ax1_r.set_ylabel("Discriminant value", color='tab:blue')
ax1_r.tick_params(axis='y',colors='tab:blue')
ax1_r.set_ylim(-1, 1.5)

# 上段凡例（右外）
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_r.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,
           loc='center left', bbox_to_anchor=(1.20, 0.5),
           frameon=False)

# -------------------------
# (2) 下段: β∥, β, ξ, B⊥²
# -------------------------
ax2 = axes[1]

beta_parallel_vals = beta_parallel_0 * r_ratios**(a)
beta_iso_vals      = beta_0 * r_ratios**(d)
xi_vals            = xi_0 * r_ratios**(b)
B_perp_vals        = B_perp_squared_0 * r_ratios**(c)

ax2.plot(r_ratios, beta_iso_vals,      lw=2, color='tab:brown',
            label=r'$\beta$ (isotropic)')
ax2.plot(r_ratios, beta_parallel_vals, lw=2, color='tab:purple',
            label=r'$\beta_{\parallel}$ (CGL)')
ax2.plot(r_ratios, xi_vals,            lw=2, color='tab:orange',
            label=r'$\xi=T_{\perp}/T_{\parallel}$')
ax2.plot(r_ratios, B_perp_vals,        lw=2, color='tab:cyan',
            label=r'$\hat{B}_{\perp}^{2}$')

ax2.set_yscale('log')
ax2.set_xlabel(r'$R/R_{0}$')
ax2.set_ylabel(r'$\beta,\;\beta_{\parallel},\;\xi,\;\hat{B}_{\perp}^{2}$')
ax2.set_ylim(1e-5, 1e2)
ax2.set_yticks([1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2])
from matplotlib.ticker import LogLocator
# ───── log‐grid 追加ここから ─────
ax2.yaxis.set_minor_locator(LogLocator(base=10.0,          # 10 のべき
                                       subs=np.arange(1, 10)*0.1,  # 1–9 の副目盛り
                                       numticks=100))
ax2.grid(True,  which='major', alpha=0.6)  # メジャー
ax2.grid(True,  which='minor', axis='y', alpha=0.3)  # マイナー
# ───── ここまで ─────
ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

# -------------------------
# 図全体レイアウト
# -------------------------
plt.subplots_adjust(right=0.8)  # 右 20 % を凡例スペースに
plt.show()

# ===============================
# 4. 各 r_ratio に対するデータセット (90×3) の作成と出力
# ===============================
# 出力するデータセットの各行は [max_imaginary_CGL, max_imaginary_isotropic, firehose_condition] となる
# ※ CGL の値は、firehose 条件が正の範囲 (r < r_ratios_cutoff) では計算済み。条件が成立しない r_ratio には NaN を補完

# 90 行のデータセットを作成
dataset = np.empty((len(r_ratios), 3))
dataset[:, 1] = max_imaginary_parts_iso          # 等方モデルの値は全 r_ratio に対してある
dataset[:, 2] = firehose_condition               # Firehose 条件

# CGL の値は、cutoff_index まで計算されているので、それ以降は NaN とする
dataset[:cutoff_index, 0] = max_imaginary_parts_CGL
dataset[cutoff_index:, 0] = np.nan

# dataset の各行に対応する r_ratio の値も表示したい場合は、別途 r_ratios を利用するか、カラムとして結合する
# 例: [r_ratio, max_imaginary_CGL, max_imaginary_isotropic, firehose_condition] の 90×4 のデータセット
# 今回は 90×3 (各モデルの値のみ) を出力する

# データセットの出力（例：標準出力に表示）
print("行: r_ratio のインデックス (1～90)")
print("列: [γₘₐₓ/ω₀ (CGL), γₘₐₓ/ω₀ (isotropic), Firehose 条件]")
print(dataset)

# -------------------------
# 追加: 下段プロット用データも含めて保存
# -------------------------
# 8 列: r, γCGL, γiso, Firehose, β∥, β, ξ, B⊥²
full_dataset = np.column_stack([
    r_ratios,
    dataset,                   # 既存の 3 列
    beta_parallel_vals,
    beta_iso_vals,
    xi_vals,
    B_perp_vals
])

# ヘッダー行を列名に合わせて作成
header = ("r_ratio, gamma_CGL, gamma_iso, firehose, "
          "beta_parallel, beta_iso, xi, B_perp_squared")

# 標準出力で確認
print("\n--- full_dataset preview ---")
print(full_dataset[:5])   # 先頭 5 行だけ表示

# CSV 保存
np.savetxt("Fig3b_dataset_full.csv", full_dataset,
           delimiter=",", header=header, comments="")
print(">> Fig3b_dataset_full.csv を保存しました")

# シナリオ３　平行方向等温で垂直方向断熱、磁場と密度はPSPスケーリング　beta=0.1　本当の完成版

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import matplotlib.ticker as mticker

# ===== ここから追加／置き換え ===================================
plt.rcParams.update({
    'font.size': 20,        # デフォルトのフォントサイズ
    'axes.titlesize': 20,   # タイトル
    'axes.labelsize': 20,   # 軸ラベル
    'xtick.labelsize': 20,  # x 目盛ラベル
    'ytick.labelsize': 20,  # y 目盛ラベル
    'legend.fontsize': 20   # 凡例
})
# ================================================================

flg = 18

a = 0.73
b = -1.66
c = 0.92

# 固定定数
B_perp_squared_0 = 0.001  # 初期 B_perp^2 の値

# CGLモデル用の初期値
beta_parallel_0 = 0.1/7  # 初期 beta_parallel_0
xi_0 = 10             # 初期 xi_0

# isotropicモデル用の定数（CGLとは別に使う）
n0 = 4e8              # 単位: cm^-3 (例)
k_b = 1.38e-16        # erg/K (例)
B0 = 10               # 初期磁場（任意単位）
#beta_0 = 0.1        # isotropic用基準 beta_0

# ------------------------------
# CGLモデルのγmax/ω0の計算関数
def calculate_max_imaginary_part_CGL(r_ratio, xi_0, beta_parallel_0):
    beta_parallel = beta_parallel_0 * r_ratio**(a)
    xi = xi_0 * r_ratio**(b)
    B_perp_squared = B_perp_squared_0 * r_ratio**(c)
    tilde_beta = beta_parallel * (3.0/2.0) / (1.0 + B_perp_squared + beta_parallel/2.0*(xi - 1.0))

    def newton_method(omega_hat, k_hat):
        f = ((omega_hat**2 - tilde_beta * k_hat**2 * (1.0 + B_perp_squared * xi / 3.0)) *
             ((omega_hat - k_hat) * ((omega_hat + k_hat)**2 - 4.0) +
              tilde_beta * B_perp_squared * (xi - 4.0) / (3.0*(1.0 + B_perp_squared)) *
              ((k_hat**2+1.0)*omega_hat + k_hat*(k_hat**2-3.0))) -
             (B_perp_squared * k_hat**2 * (1.0 - tilde_beta*(3.0 - xi - B_perp_squared) / (3.0*(1.0+B_perp_squared))) *
              ((omega_hat**3+omega_hat**2*k_hat-3.0*omega_hat+k_hat) - tilde_beta*(3.0 - xi)/3.0 *
               ((k_hat**2+1.0)*omega_hat + k_hat*(k_hat**2-3.0)))))
        df = (2 * omega_hat * ((omega_hat - k_hat)*((omega_hat+k_hat)**2 - 4.0) +
                tilde_beta*B_perp_squared*(xi-4.0)/(3.0*(1.0+B_perp_squared))*((k_hat**2+1.0)*omega_hat+k_hat*(k_hat**2-3.0))) +
              (omega_hat**2 - tilde_beta*k_hat**2*(1.0+B_perp_squared*xi/3.0))*
              (((omega_hat+k_hat)**2-4.0)+(omega_hat-k_hat)*2*(omega_hat+k_hat)) +
              tilde_beta*B_perp_squared*(xi-4.0)/(3.0*(1.0+B_perp_squared))*(k_hat**2+1.0) -
              B_perp_squared*k_hat**2*(1.0-tilde_beta*(3.0-xi-B_perp_squared)/(3.0*(1.0+B_perp_squared)))*
              (3*omega_hat**2+2*omega_hat*k_hat-3.0) -
              (B_perp_squared*k_hat**2*(-tilde_beta*(3.0-xi)/3.0))*(k_hat**2+1.0) )
        if np.abs(df) < 1e-10:
            return omega_hat
        return omega_hat - f/df

    def root_of_func(initial_guesses, k_hat):
        roots = []
        for omega_hat in initial_guesses:
            for _ in range(300):
                omega_hat_new = newton_method(omega_hat, k_hat)
                if np.abs(omega_hat_new - omega_hat) < 1e-10:
                    if all(np.abs(omega_hat_new - root) > 1e-10 for root in roots):
                        roots.append(omega_hat_new)
                    break
                omega_hat = omega_hat_new
        return roots

    k_hat_values = np.linspace(0.85, 2.5, 1000)
    initial_guesses = [0.1+0.2j, 2+0.5j, -5+2j, -5-8j, 7-3j, 1+1j, 2+0.1j, 2-0.1j,
                       -2+2j, -2-2j, 3+3j, 3-3j, -4+0.5j, 4+0.5j, 0+3j, 0-3j,
                       1+2j, 1-2j, -1+0.5j, -1-0.5j, 0.5+1j, 0.5-1j, -0.5+0.2j,
                       -0.5-0.2j, 3.5+4j, 3.5-4j, -3.5+4j, -3.5-4j,
                       2.15+0.1j, 2.20+0.1j, 2.25+0.1j, 2.1+0.1j, 2.15+0.07j, 2.15+0.04j, 2.15+0.01j, -4-4j,
                       6+2j, 6-2j, -6+2j, -6-2j, 0.1+0.3j, 0.1-0.3j, -0.1+0.3j, -0.1-0.3j]
    omega_hats_imag = []
    for k_hat in k_hat_values:
        roots = root_of_func(initial_guesses, k_hat)
        for root in roots:
            omega_hats_imag.append(root.imag)
    return np.max(omega_hats_imag)

# ------------------------------
# isotropicモデルのγmax/ω0の計算関数
def calculate_max_imaginary_part_isotropic(args):
    r_ratio, B_perp_squared_0 = args
    # 新たな β の計算（density, B, T のスケーリングに基づく）
    n = n0 * r_ratio**(-2.59)
    B = B0 * r_ratio**(-1.66)
    T = 1.03e6 * (1 + 20 * r_ratio**(-1.66)) / 3
    beta = n * k_b * T * 8 * np.pi / (B**2)

    B_perp_squared = B_perp_squared_0 * r_ratio**(c)

    def newton_method(omega_hat, k_hat):
        f = (omega_hat - k_hat) * (omega_hat**2 - beta*k_hat**2) * ((omega_hat+k_hat)**2 - 4) \
            - (B_perp_squared * k_hat**2) * (omega_hat**3 + k_hat*omega_hat**2 - 3*omega_hat + k_hat)
        df = (3 * B_perp_squared * k_hat**2 + beta*k_hat**4 + 4*beta*k_hat**2 +
              4*k_hat*omega_hat**3 + 5*omega_hat**4 +
              3*omega_hat**2 * (-B_perp_squared*k_hat**2 - beta*k_hat**2 - k_hat**2 - 4) +
              2*omega_hat * (-B_perp_squared*k_hat**3 - beta*k_hat**3 - k_hat**3 + 4*k_hat))
        if np.abs(df) < 1e-10:
            return omega_hat
        return omega_hat - f/df

    def root_of_func(initial_guesses, k_hat):
        roots = []
        for omega_hat in initial_guesses:
            for _ in range(300):
                omega_hat_new = newton_method(omega_hat, k_hat)
                if np.abs(omega_hat_new - omega_hat) < 1e-10:
                    if all(np.abs(omega_hat_new - root) > 1e-10 for root in roots):
                        roots.append(omega_hat_new)
                    break
                omega_hat = omega_hat_new
        return roots

    k_hat_values = np.linspace(0.85, 2.5, 1000)
    initial_guesses = [0.1+0.2j, 2+0.5j, -5+2j, -5-8j, 7-3j, 1+1j, 2+0.1j, 2-0.1j,
                       -2+2j, -2-2j, 3+3j, 3-3j, -4+0.5j, 4+0.5j, 0+3j, 0-3j,
                       1+2j, 1-2j, -1+0.5j, -1-0.5j, 0.5+1j, 0.5-1j, -0.5+0.2j,
                       -0.5-0.2j, 3.5+4j, 3.5-4j, -3.5+4j, -3.5-4j,
                       2.15+0.1j, 2.20+0.1j, 2.25+0.1j, 2.1+0.1j, 2.15+0.07j, 2.15+0.04j, 2.15+0.01j, -4-4j,
                       6+2j, 6-2j, -6+2j, -6-2j, 0.1+0.3j, 0.1-0.3j, -0.1+0.3j, -0.1-0.3j]
    omega_hats_imag = []
    for k_hat in k_hat_values:
        roots = root_of_func(initial_guesses, k_hat)
        for root in roots:
            omega_hats_imag.append(root.imag)
    return np.max(omega_hats_imag)

# ------------------------------
# Firehose条件（CGLモデル用）の計算関数
def calculate_firehose_condition(r_ratios, xi_0, beta_parallel_0, B_perp_squared_0):
    # xi, beta_parallel, B_perp_squared のスケーリング（CGLモデル）
    xi_values = xi_0 * r_ratios**(b)
    beta_parallel_values = beta_parallel_0 * r_ratios**(a)
    B_perp_squared_values = B_perp_squared_0 * r_ratios**(c)
    condition = 1 + 0.5 * beta_parallel_values * (xi_values - 1) / (1 + B_perp_squared_values)
    return condition

# ------------------------------
# r_ratio の範囲
r_ratios = np.linspace(1, 30, 90)

# (A) CGLモデルのγmax/ω0の計算（Firehose条件が正となる領域のみ）
firehose_condition = calculate_firehose_condition(r_ratios, xi_0, beta_parallel_0, B_perp_squared_0)
cutoff_index = np.argmax(firehose_condition < 0) if np.any(firehose_condition < 0) else len(r_ratios)
r_ratios_cutoff = r_ratios[:cutoff_index]

with Pool() as pool:
    max_imaginary_parts_CGL = pool.starmap(
        calculate_max_imaginary_part_CGL,
        [(r, xi_0, beta_parallel_0) for r in r_ratios_cutoff]
    )
    max_imaginary_parts_CGL = [result for result in max_imaginary_parts_CGL]

# (B) isotropicモデルのγmax/ω0の計算（全r_ratio）
with Pool() as pool:
    max_imaginary_parts_iso = pool.map(
        calculate_max_imaginary_part_isotropic,
        [(r, B_perp_squared_0) for r in r_ratios]
    )
    max_imaginary_parts_iso = [result for result in max_imaginary_parts_iso]

# =========================================================
# 連結プロット：上段を下段より少し大きく
# =========================================================
fig, axes = plt.subplots(
    nrows=2, ncols=1, sharex=True, figsize=(15, 10),
    gridspec_kw={
        'hspace': 0.05,           # パネル間の縦スペース
        'height_ratios': [1.7, 1] # ← ここで比率を指定（[上段, 下段]）
    })

# -------------------------
# (1) 上段: γmax / ω0 & Firehose
# -------------------------
ax1  = axes[0]
ax1_r = ax1.twinx()

ax1.plot(r_ratios_cutoff, max_imaginary_parts_CGL,
         color='tab:red', lw=2, label=r"$\gamma_{max}/\omega_{0}$ (CGL)")
ax1.plot(r_ratios, max_imaginary_parts_iso,
         color='tab:green', lw=2, label=r"$\gamma_{max}/\omega_{0}$ (isotropic)")
ax1_r.plot(r_ratios, firehose_condition,
           color='tab:blue', lw=2, label="Discriminant value")

ax1.set_ylabel(r"$\gamma_{max}/\omega_{0}$")
ax1.tick_params(axis='y')
ax1.grid(True, alpha=0.3)
ax1_r.set_ylabel("Discriminant value", color='tab:blue')
ax1_r.tick_params(axis='y',colors='tab:blue')
ax1_r.set_ylim(-1, 1.5)

# 上段凡例（右外）
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_r.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,
           loc='center left', bbox_to_anchor=(1.20, 0.5),
           frameon=False)

# -------------------------
# (2) 下段: β∥, β, ξ, B⊥²
# -------------------------
ax2 = axes[1]

beta_parallel_vals = beta_parallel_0 * r_ratios**(a)
beta_iso_vals      = beta_0 * r_ratios**(d)
xi_vals            = xi_0 * r_ratios**(b)
B_perp_vals        = B_perp_squared_0 * r_ratios**(c)

ax2.plot(r_ratios, beta_iso_vals,      lw=2, color='tab:brown',
            label=r'$\beta$ (isotropic)')
ax2.plot(r_ratios, beta_parallel_vals, lw=2, color='tab:purple',
            label=r'$\beta_{\parallel}$ (CGL)')
ax2.plot(r_ratios, xi_vals,            lw=2, color='tab:orange',
            label=r'$\xi=T_{\perp}/T_{\parallel}$')
ax2.plot(r_ratios, B_perp_vals,        lw=2, color='tab:cyan',
            label=r'$\hat{B}_{\perp}^{2}$')

ax2.set_yscale('log')
ax2.set_xlabel(r'$R/R_{0}$')
ax2.set_ylabel(r'$\beta,\;\beta_{\parallel},\;\xi,\;\hat{B}_{\perp}^{2}$')
ax2.set_ylim(1e-5, 1e2)
ax2.set_yticks([1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2])
from matplotlib.ticker import LogLocator
# ───── log‐grid 追加ここから ─────
ax2.yaxis.set_minor_locator(LogLocator(base=10.0,          # 10 のべき
                                       subs=np.arange(1, 10)*0.1,  # 1–9 の副目盛り
                                       numticks=100))
ax2.grid(True,  which='major', alpha=0.6)  # メジャー
ax2.grid(True,  which='minor', axis='y', alpha=0.3)  # マイナー
# ───── ここまで ─────
ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

# -------------------------
# 図全体レイアウト
# -------------------------
plt.subplots_adjust(right=0.8)  # 右 20 % を凡例スペースに
plt.show()

# ===============================
# 4. 各 r_ratio に対するデータセット (90×3) の作成と出力
# ===============================
# 出力するデータセットの各行は [max_imaginary_CGL, max_imaginary_isotropic, firehose_condition] となる
# ※ CGL の値は、firehose 条件が正の範囲 (r < r_ratios_cutoff) では計算済み。条件が成立しない r_ratio には NaN を補完

# 90 行のデータセットを作成
dataset = np.empty((len(r_ratios), 3))
dataset[:, 1] = max_imaginary_parts_iso          # 等方モデルの値は全 r_ratio に対してある
dataset[:, 2] = firehose_condition               # Firehose 条件

# CGL の値は、cutoff_index まで計算されているので、それ以降は NaN とする
dataset[:cutoff_index, 0] = max_imaginary_parts_CGL
dataset[cutoff_index:, 0] = np.nan

# dataset の各行に対応する r_ratio の値も表示したい場合は、別途 r_ratios を利用するか、カラムとして結合する
# 例: [r_ratio, max_imaginary_CGL, max_imaginary_isotropic, firehose_condition] の 90×4 のデータセット
# 今回は 90×3 (各モデルの値のみ) を出力する

# データセットの出力（例：標準出力に表示）
print("行: r_ratio のインデックス (1～90)")
print("列: [γₘₐₓ/ω₀ (CGL), γₘₐₓ/ω₀ (isotropic), Firehose 条件]")
print(dataset)

# -------------------------
# 追加: 下段プロット用データも含めて保存
# -------------------------
# 8 列: r, γCGL, γiso, Firehose, β∥, β, ξ, B⊥²
full_dataset = np.column_stack([
    r_ratios,
    dataset,                   # 既存の 3 列
    beta_parallel_vals,
    beta_iso_vals,
    xi_vals,
    B_perp_vals
])

# ヘッダー行を列名に合わせて作成
header = ("r_ratio, gamma_CGL, gamma_iso, firehose, "
          "beta_parallel, beta_iso, xi, B_perp_squared")

# 標準出力で確認
print("\n--- full_dataset preview ---")
print(full_dataset[:5])   # 先頭 5 行だけ表示

# CSV 保存
np.savetxt("Fig3c_dataset_full.csv", full_dataset,
           delimiter=",", header=header, comments="")
print(">> Fig3c_dataset_full.csv を保存しました")

#シナリオ４＆５　beta=0.001

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
from io import StringIO
from multiprocessing import Pool
import sympy as sp

#############################
# ① βₚₐᵣₐ の計算（Tₚₐᵣₐ データから）
#############################

# --- Tₚₐᵣₐ のデータ定義 ---
data_str_Tpara = """
0.9363634313172465	630950.7344480193
0.9518215986713207	735640.22544596407
0.952918054833259	874310.24580220731
0.9679063371940468	908510.75756516862
0.998210703186863	926110.87281287937
1.0136520170010803	926110.87281287937
1.110736672952074	841390.51416451947
1.2356312656535287	735640.22544596407
1.3529376348534548	595660.214352901094
1.5242477252204125	348070.00588428406
1.5691579687708173	271220.725793320296
1.6153914420970539	211340.89039836648
1.6640511310707182	181270.308811846968
1.7680722177343189	161550.98098439873
1.820865124259488	133350.21432163324
2.0274233521107727	133350.21432163324
2.294386965053555	152520.229565390198
2.4434305526939717	192010.41938638801
2.6014903677573367	232630.05067153624
2.856128729537577	281830.82931264455
3.0404960563039145	334960.54391578279
3.3897421734448465	405810.98942243799
3.665782272509721	510890.69774506924
3.9627803425543924	607200.21956909884
4.6977390496362625	735640.22544596407
5.484172555736265	891250.0938133744
6.1125471512033736	1039120.2303835169
7.241580008400507	1143750.58630495366
8.447383717827934	1234990.91726875826
10.966184681342897	1234990.91726875826
12.58764378600493	1188500.22274370166
15.127869790847797	1122010.84543019629
18.180721372437116	1059250.37251772876
21.18632136291081	980990.47127268763
25.471560675002788	980990.47127268763
30.15705228022852	980990.47127268763
1.3518996152492362	530880.44442309879
1.4147075349881577	482310.78482239309
1.4580679970023382	446680.35921509626
1.4796761259208662	405810.98942243799
1.523078270709203	310210.776647994982
1.5679540575987572	241730.154808041014
1.2539429491545022	668340.39175686135
1.766941707999398	146770.992676220705
2.1566406048857756	141250.375446227554
2.3680392085234327	174440.826989992558
2.4828154797631	211340.89039836648
2.6853444456585054	271220.725793320296
2.993030283874072	316220.776601683792
3.235108784158188	368690.450645195735
3.4443804148047565	446680.35921509626
4.152726538879247	681290.20690579608
4.922913696391965	825400.41852680173
9.405651624378091	1234990.91726875826
6.603554671772791	1122010.84543019629
5.745572820257177	962350.06263980868
13.59182511235687	1188500.22274370166
16.33260863538179	1100690.4171252208
22.87939016505923	1000000
28.364672776598606	1000000
19.0399878392478	1079770.51623277094
19.3295713029763	1039120.2303835169
7.821278990952227	1188500.22274370166
1.1449269489307676	794320.82347242805
32.070897303318624	1000000
34.62492878921517	980990.47127268763
37.965479460384884	980990.47127268763
42.26686142656028	962350.06263980868
47.04354057259765	908510.75756516862
"""

# --- 文字列からデータを読み込み ---
data_Tpara = np.genfromtxt(data_str_Tpara.strip().splitlines())
R_Tpara = data_Tpara[:, 0]         # R/Rsun
Tpara = data_Tpara[:, 1]           # T_para

# --- R の昇順にソート ---
idx = np.argsort(R_Tpara)
R_Tpara = R_Tpara[idx]
Tpara = Tpara[idx]

# --- R 軸の対数変換 ---
log_R_Tpara = np.log(R_Tpara)

# --- スプライン補間（smoothing_parameter は適宜調整） ---
smoothing_parameter = 1e11
spline_Tpara = UnivariateSpline(log_R_Tpara, Tpara, s=smoothing_parameter)

# --- R/Rsun = 1 から 30 の範囲を 60 分割 ---
R_target = np.linspace(1, 30, 60)

# --- 対数変換してスプライン関数で T_para を評価 ---
T_target = spline_Tpara(np.log(R_target))

k_b = 1.38e-16       # erg/K
m_p = 1.6726219e-24  # g

# n(R) と B(R) の定義（CGS単位系）
n_0 = 3e7    # Rsun=1 における数密度 [cm^-3]
B_0 = 10     # Rsun=1 における磁場 [G]

# --- Rsun=1 における T_para ---
Tpara_0 = spline_Tpara(np.log(1))
print("Rsun=1 のときの T_para =", Tpara_0)

# --- R に依存する n(R) と B(R) の定義 ---
n_R = n_0 * (R_target)**(-2.59)
B_R = B_0 * (R_target)**(-1.66)

# --- β_parallel の計算 ---
beta_para = 8 * np.pi * n_R * k_b * T_target / (B_R**2)

# --- β_parallel のプロット（参考） ---
plt.figure(figsize=(10,6))
plt.plot(R_target, beta_para, 'b-', lw=2, label=r'$\beta_{\parallel}$')
plt.xlabel(r'$R/R_\odot$')
plt.ylabel(r'$\beta_{\parallel}$')
plt.title(r'$\beta_{\parallel}$ vs $R/R_\odot$')
plt.legend()
plt.grid(True)
plt.show()

# --- 補間関数を作成（PDI 用に β_parallel を任意の r で評価） ---
beta_para_interp = UnivariateSpline(R_target, beta_para, s=0)

#############################
# ② T_perp/T_para（=xi）のバンプモデルフィッティング
#############################

# --- T_perp/T_para のデータ定義 ---
data_str_bump = """
1.0009813481265506	0.910543130990412
1.0810468494787815	1.0894568690095827
1.149562217230484	1.1341853035143732
1.2413904876306563	1.2236421725239612
1.2998954188768872	1.2236421725239612
1.3611576025865708	1.2236421725239612
1.4036636255995907	1.2683706070287517
1.4698162542259607	1.2683706070287517
1.5396905343022556	1.6261980830670897
1.6373544520272827	1.7156549520766742
1.6630904432580889	1.9392971246006354
1.715782341439801	2.3865814696485597
1.7428365372600465	2.654952076677315
1.7436059778765893	3.057507987220447
1.7988488596018384	3.5047923322683694
1.7992900215245589	3.7284345047923306
1.8003492516192892	4.265175718849839
1.8288265752326527	4.578274760383385
1.8578454563701159	4.936102236421723
1.858665671991507	5.338658146964855
1.917459963137651	5.741214057507985
1.9184005802972979	6.1884984025559095
1.9191534062389042	6.546325878594249
1.9494143261746528	6.814696485623003
1.9804437570384015	7.217252396166133
1.9811237668749409	7.5303514376996805
1.9821928255364416	8.022364217252395
2.014040305942652	8.559105431309902
2.01483066318364	8.916932907348244
2.0463994732015016	9.095846645367413
2.047202528948116	9.45367412140575
2.0483072451864515	9.945686900958465
2.0810128425639136	10.39297124600639
2.0816252916659104	10.661341853035143
2.082237921013744	10.929712460063897
2.0828507306604624	11.19808306709265
2.1820823652270755	11.645367412140576
2.183045727403937	12.047923322683706
2.1836882051282536	12.316293929712462
2.1846522762634235	12.718849840255592
2.185831163457327	13.210862619808307
2.220623714253556	13.613418530351439
2.255748799054423	13.926517571884984
2.255970068843575	14.015974440894569
2.435343187173589	13.792332268370608
2.47240914208274	13.568690095846645
2.509916143602259	13.300319488817891
2.547992134751788	13.031948881789138
2.547742222888948	12.942492012779553
2.5472424726963405	12.763578274760384
2.585884712634639	12.49520766773163
2.585123901946731	12.226837060702875
2.5848703481302975	12.13738019169329
2.6649367198252545	11.95846645367412
2.7057624775554308	11.824281150159745
2.7049663968168707	11.555910543130992
2.788889517327686	11.421725239616613
2.8313364519693063	11.19808306709265
2.874711386608766	11.063897763578275
2.9184645312144477	10.840255591054312
2.9630289125374594	10.661341853035143
3.0545091262274338	10.39297124600639
3.054059749243192	10.258785942492011
3.3449347197682875	9.230031948881788
3.1006946429665376	10.079872204472842
3.246029801024168	9.856230031948881
3.2453930788300234	9.677316293929712
3.2951111943071494	9.543130990415335
3.5020613401114877	9.095846645367413
3.447698552067784	8.827476038338656
3.4996576261267096	8.46964856230032
3.6080595382385616	8.29073482428115
3.662614979821021	7.977635782747603
3.775508965117988	7.6645367412140555
3.891882718828082	7.3514376996805115
4.073902675725465	7.038338658146964
4.072304606697792	6.680511182108624
4.26255368274972	6.322683706070286
4.462128429352407	6.054313099041533
4.529597766895492	5.741214057507985
4.6703601576246125	5.651757188498401
4.814552319712819	5.383386581469647
4.887350460940839	5.070287539936102
5.116680337023466	4.891373801916931
5.43963494466012	4.712460063897762
5.522426259101111	4.488817891373801
5.870990736112753	4.309904153354632
6.052547858444415	4.08626198083067
6.434256983067591	3.8626198083067074
6.734851256908489	3.5047923322683694
7.27176846857922	3.460063897763577
7.970598769617529	3.1469648562300296
8.475349463592782	3.1469648562300296
8.60518840547276	3.012779552715653
9.877171346548483	2.7444089456868994
9.433550667216632	2.833865814696484
10.664078238599888	2.654952076677315
11.513112761822988	2.520766773162938
12.055117812022477	2.476038338658146
12.241598338236543	2.476038338658146
13.837733513610797	2.252396166134183
13.420670934728445	2.341853035143769
15.887064673045423	2.2076677316293907
16.633351518195692	2.0734824281150157
19.095769650509038	1.9840255591054294
19.99474491481308	1.9392971246006354
20.93501466425129	1.8498402555910527
20.93501466425129	1.8498402555910527
22.259665654857805	1.8051118210862622
25.163252361956864	1.6261980830670897
25.163252361956864	1.6261980830670897
25.94777399049993	1.6261980830670897
29.33100497427884	1.4025559105431284
29.33100497427884	1.4025559105431284
32.15765303275845	1.3130990415335457
"""

# 文字列から数値データを読み込む
data_array_bump = np.genfromtxt(StringIO(data_str_bump))
R_bump = data_array_bump[:, 0]  # R/Rsun
xi_data = data_array_bump[:, 1] # T_perp/T_para

# バンプモデルの定義（対数座標上のシグモイド）
def bump_model(x, A, D, k1, mu1, k2, mu2):
    logistic_rise = 1.0 / (1 + np.exp(-k1 * (np.log(x) - mu1)))
    logistic_fall = 1.0 / (1 + np.exp(-k2 * (np.log(x) - mu2)))
    return A + D * logistic_rise * (1 - logistic_fall)

# 初期推定値とパラメータ境界
initial_guess = [1.3, 13.0, 10.0, 0.3, 10.0, 1.0]
lower_bounds = [0.0,   0.0,  0.0, -5.0,  0.0, -5.0]
upper_bounds = [10.0,  50.0, 50.0,  5.0, 50.0,  5.0]

# フィッティング実行
popt, pcov = curve_fit(
    bump_model, R_bump, xi_data,
    p0=initial_guess,
    bounds=(lower_bounds, upper_bounds),
    maxfev=10000
)

print("バンプモデルの最適化パラメータ:")
print("A =", popt[0])
print("D =", popt[1])
print("k1 =", popt[2])
print("mu1 =", popt[3])
print("k2 =", popt[4])
print("mu2 =", popt[5])

# フィッティング結果のプロット（参考）
x_fit = np.linspace(np.min(R_bump), np.max(R_bump), 1000)
y_fit = bump_model(x_fit, *popt)
plt.figure(figsize=(8,5))
plt.scatter(R_bump, xi_data, color='red', label='Data')
plt.plot(x_fit, y_fit, label='Fitted Bump Model')
plt.xscale("log")
plt.xlabel("R/Rsun")
plt.ylabel("T_perp/T_para")
plt.title("Fitted Bump Model")
plt.legend()
plt.grid(True)
plt.show()

#############################
# ③ PDI 成長率と Firehose 条件の計算・プロット
#############################

# 固定定数（r=1 における B_perp² の値）
B_perp_squared_0 = 0.001

# ※ ここでは T_perp/T_para = xi をバンプモデルから得るので，
#    固定値 xi0 は不要となります。

def calculate_max_imaginary_part(r_ratio):
    # beta_parallel は補間関数から取得
    beta_parallel = beta_para_interp(r_ratio)
    # xi を固定のスケーリングではなく，バンプモデルから求める．
    xi = bump_model(r_ratio, *popt)
    # ※ B_perp_squared の r によるスケーリングはここでは例として r^1 を用いる（必要に応じて調整）
    B_perp_squared = B_perp_squared_0 * r_ratio**(0.92)
    tilde_beta = beta_parallel * (3.0/2.0) / (1.0 + B_perp_squared + beta_parallel/2.0 * (xi - 1.0))

    def newton_method(omega_hat, k_hat):
        f = ((omega_hat**2 - tilde_beta * k_hat**2 * (1.0 + B_perp_squared * xi / 3.0)) *
             ((omega_hat - k_hat) * ((omega_hat + k_hat)**2 - 4.0) +
              tilde_beta * B_perp_squared * (xi - 4.0) / (3.0*(1.0+B_perp_squared)) *
             ((k_hat**2+1.0)*omega_hat + k_hat*(k_hat**2-3.0))) -
             (B_perp_squared * k_hat**2 * (1.0 - tilde_beta*(3.0-xi-B_perp_squared) / (3.0*(1.0+B_perp_squared))) *
              ((omega_hat**3+omega_hat**2*k_hat-3.0*omega_hat+k_hat) -
               tilde_beta*(3.0-xi)/3.0 * ((k_hat**2+1.0)*omega_hat+k_hat*(k_hat**2-3.0)))))
        df = (2*omega_hat * (((omega_hat-k_hat)*((omega_hat+k_hat)**2-4.0)) +
              tilde_beta*B_perp_squared*(xi-4.0)/(3.0*(1.0+B_perp_squared)) *
              ((k_hat**2+1.0)*omega_hat+k_hat*(k_hat**2-3.0))) +
              (omega_hat**2 - tilde_beta*k_hat**2*(1.0+B_perp_squared*xi/3.0)) *
              (((omega_hat+k_hat)**2-4.0) + (omega_hat-k_hat)*2*(omega_hat+k_hat)) +
              tilde_beta*B_perp_squared*(xi-4.0)/(3.0*(1.0+B_perp_squared))*(k_hat**2+1.0) -
              B_perp_squared*k_hat**2*(1.0 - tilde_beta*(3.0-xi-B_perp_squared) / (3.0*(1.0+B_perp_squared))) *
              (3*omega_hat**2+2*omega_hat*k_hat-3.0) -
              (B_perp_squared*k_hat**2 * (-tilde_beta*(3.0-xi)/3.0))*(k_hat**2+1.0)
             )
        if np.abs(df) < 1e-10:
            return omega_hat
        return omega_hat - f/df

    def root_of_func(initial_guesses, k_hat):
        roots = []
        for omega_hat in initial_guesses:
            for _ in range(1000):
                omega_hat_new = newton_method(omega_hat, k_hat)
                if np.abs(omega_hat_new - omega_hat) < 1e-10:
                    if all(np.abs(omega_hat_new - root) > 1e-10 for root in roots):
                        roots.append(omega_hat_new)
                    break
                omega_hat = omega_hat_new
        return roots

    # k_hat の範囲を決め，各 k_hat での根の虚部の最大値を求める
    k_hat_values = np.linspace(0.95, 2, 100)
    omega_hats_imag = []
    initial_guesses = [0.1+0.2j, 2+0.5j, -5+2j, -5-8j, 7-3j, 1+1j, 2+0.1j, 2-0.1j,
                       -2+2j, -2-2j, 3+3j, 3-3j, -4+0.5j, 4+0.5j, 0+3j, 0-3j,
                       1+2j, 1-2j, -1+0.5j, -1-0.5j, 0.5+1j, 0.5-1j, -0.5+0.2j,
                       -0.5-0.2j, 3.5+4j, 3.5-4j, -3.5+4j, -3.5-4j,
                       2.15+0.1j, 2.20+0.1j, 2.25+0.1j, 2.1+0.1j, 2.15+0.07j, 2.15+0.04j, 2.15+0.01j,
                       -4-4j, 6+2j, 6-2j, -6+2j, -6-2j, 0.1+0.3j, 0.1-0.3j, -0.1+0.3j, -0.1-0.3j]
    for k_hat in k_hat_values:
        roots = root_of_func(initial_guesses, k_hat)
        for root in roots:
            omega_hats_imag.append(root.imag)
    return np.max(omega_hats_imag)

def calculate_firehose_condition(r_ratios):
    beta_parallel_values = beta_para_interp(r_ratios)
    # xi を各 r で，バンプモデルから求める
    xi_values = bump_model(r_ratios, *popt)
    B_perp_squared_values = B_perp_squared_0 * r_ratios**(0.92)
    condition = 1 + 0.5 * beta_parallel_values * (xi_values - 1) / (1 + B_perp_squared_values)
    return condition

# r_ratio の範囲（1～30）
r_ratios = np.linspace(1, 30, 90)

# Firehose 条件の計算
firehose_condition = calculate_firehose_condition(r_ratios)
cutoff_index = np.argmax(firehose_condition < 0) if np.any(firehose_condition < 0) else len(r_ratios)
r_ratios_cutoff = r_ratios[:cutoff_index]

# 並列計算で各 r_ratio に対する最大虚部を求める
with Pool() as pool:
    max_imaginary_parts = pool.map(calculate_max_imaginary_part, r_ratios_cutoff)

# プロット①：PDI 成長率（最大虚部） vs r_ratio
plt.figure(figsize=(14,8))
plt.plot(r_ratios_cutoff, max_imaginary_parts, lw=2, color='tab:blue',
         label=f"$\\beta_{{||}}(r=1) = {beta_para_interp(1):.3e}$, $\\xi(r=1) = {bump_model(1, *popt):.3f}$")
plt.xlabel(r"$r/r_{0}$", fontsize=20)
plt.ylabel(r"$\gamma/\omega_{0}$", fontsize=20)
plt.title("Maximum Growth Rate (PDI) vs r", fontsize=22)
plt.legend(fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.show()

# プロット②：Firehose 条件 vs r_ratio
plt.figure(figsize=(14,8))
plt.plot(r_ratios, firehose_condition, lw=2, color='tab:green',
         label=f"$\\xi(r=1) = {bump_model(1, *popt):.3f}$, $\\beta_{{||}}(r=1) = {beta_para_interp(1):.3e}$")
plt.axhline(0, color='black', lw=1, ls='--')
plt.xlabel(r"$r/r_{0}$", fontsize=20)
plt.ylabel("Discriminant Value", fontsize=20)
plt.title("Firehose Condition vs r", fontsize=22)
plt.xlim(1, 30)
plt.ylim(-1, 1.5)
plt.legend(fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
from io import StringIO
from multiprocessing import Pool
import sympy as sp

# フォントサイズの設定
plt.rcParams.update({'font.size': 16})

#############################################
# ① T_parallel のスプライン補間によるフィッティング
#############################################
data_str_Tpara = """
0.9363634313172465	630950.7344480193
0.9518215986713207	735640.22544596407
0.952918054833259	874310.24580220731
0.9679063371940468	908510.75756516862
0.998210703186863	926110.87281287937
1.0136520170010803	926110.87281287937
1.110736672952074	841390.51416451947
1.2356312656535287	735640.22544596407
1.3529376348534548	595660.214352901094
1.5242477252204125	348070.00588428406
1.5691579687708173	271220.725793320296
1.6153914420970539	211340.89039836648
1.6640511310707182	181270.308811846968
1.7680722177343189	161550.98098439873
1.820865124259488	133350.21432163324
2.0274233521107727	133350.21432163324
2.294386965053555	152520.229565390198
2.4434305526939717	192010.41938638801
2.6014903677573367	232630.05067153624
2.856128729537577	281830.82931264455
3.0404960563039145	334960.54391578279
3.3897421734448465	405810.98942243799
3.665782272509721	510890.69774506924
3.9627803425543924	607200.21956909884
4.6977390496362625	735640.22544596407
5.484172555736265	891250.0938133744
6.1125471512033736	1039120.2303835169
7.241580008400507	1143750.58630495366
8.447383717827934	1234990.91726875826
10.966184681342897	1234990.91726875826
12.58764378600493	1188500.22274370166
15.127869790847797	1122010.84543019629
18.180721372437116	1059250.37251772876
21.18632136291081	980990.47127268763
25.471560675002788	980990.47127268763
30.15705228022852	980990.47127268763
1.3518996152492362	530880.44442309879
1.4147075349881577	482310.78482239309
1.4580679970023382	446680.35921509626
1.4796761259208662	405810.98942243799
1.523078270709203	310210.776647994982
1.5679540575987572	241730.154808041014
1.2539429491545022	668340.39175686135
1.766941707999398	146770.992676220705
2.1566406048857756	141250.375446227554
2.3680392085234327	174440.826989992558
2.4828154797631	211340.89039836648
2.6853444456585054	271220.725793320296
2.993030283874072	316220.776601683792
3.235108784158188	368690.450645195735
3.4443804148047565	446680.35921509626
4.152726538879247	681290.20690579608
4.922913696391965	825400.41852680173
9.405651624378091	1234990.91726875826
6.603554671772791	1122010.84543019629
5.745572820257177	962350.06263980868
13.59182511235687	1188500.22274370166
16.33260863538179	1100690.4171252208
22.87939016505923	1000000
28.364672776598606	1000000
19.0399878392478	1079770.51623277094
19.3295713029763	1039120.2303835169
7.821278990952227	1188500.22274370166
1.1449269489307676	794320.82347242805
32.070897303318624	1000000
34.62492878921517	980990.47127268763
37.965479460384884	980990.47127268763
42.26686142656028	962350.06263980868
47.04354057259765	908510.75756516862
"""
# データ読み込み
data_Tpara = np.genfromtxt(StringIO(data_str_Tpara))
R_Tpara = data_Tpara[:, 0]         # R/Rsun
Tpara = data_Tpara[:, 1]           # T_parallel [K]
idx = np.argsort(R_Tpara)
R_Tpara = R_Tpara[idx]
Tpara = Tpara[idx]
log_R_Tpara = np.log(R_Tpara)
smoothing_parameter = 1e11
spline_Tpara = UnivariateSpline(log_R_Tpara, Tpara, s=smoothing_parameter)
R_target = np.linspace(1, 30, 60)
Tpara_target = spline_Tpara(np.log(R_target))
print("R/Rsun (60点):")
print(R_target)
print("\n対応する T_parallel:")
print(Tpara_target)
plt.figure(figsize=(10, 6))
plt.plot(R_Tpara, Tpara, 'ro', label='data')
plt.plot(R_target, Tpara_target, 'gs', label='evaluated points')
plt.plot(np.exp(np.linspace(log_R_Tpara.min(), log_R_Tpara.max(), 2000)),
         spline_Tpara(np.linspace(log_R_Tpara.min(), log_R_Tpara.max(), 2000)),
         'b-', lw=2, label='smooth fitting spline')
plt.xlabel('R/Rsun (log scale)')
plt.ylabel('T_parallel')
plt.ylim(0,2000000)
plt.title('Smooth fitting spline for R/Rsun and T_parallel\n(評価点: R/Rsun=[1, 30] 60点)')
plt.legend()
plt.grid(True, which='both', ls='--')
plt.show()

#############################################
# ② T_perp/T_parallel のバンプモデルによるフィッティング
#############################################
data_str_bump = """
1.0009813481265506	0.910543130990412
1.0810468494787815	1.0894568690095827
1.149562217230484	1.1341853035143732
1.2413904876306563	1.2236421725239612
1.2998954188768872	1.2236421725239612
1.3611576025865708	1.2236421725239612
1.4036636255995907	1.2683706070287517
1.4698162542259607	1.2683706070287517
1.5396905343022556	1.6261980830670897
1.6373544520272827	1.7156549520766742
1.6630904432580889	1.9392971246006354
1.715782341439801	2.3865814696485597
1.7428365372600465	2.654952076677315
1.7436059778765893	3.057507987220447
1.7988488596018384	3.5047923322683694
1.7992900215245589	3.7284345047923306
1.8003492516192892	4.265175718849839
1.8288265752326527	4.578274760383385
1.8578454563701159	4.936102236421723
1.858665671991507	5.338658146964855
1.917459963137651	5.741214057507985
1.9184005802972979	6.1884984025559095
1.9191534062389042	6.546325878594249
1.9494143261746528	6.814696485623003
1.9804437570384015	7.217252396166133
1.9811237668749409	7.5303514376996805
1.9821928255364416	8.022364217252395
2.014040305942652	8.559105431309902
2.01483066318364	8.916932907348244
2.0463994732015016	9.095846645367413
2.047202528948116	9.45367412140575
2.0483072451864515	9.945686900958465
2.0810128425639136	10.39297124600639
2.0816252916659104	10.661341853035143
2.082237921013744	10.929712460063897
2.0828507306604624	11.19808306709265
2.1820823652270755	11.645367412140576
2.183045727403937	12.047923322683706
2.1836882051282536	12.316293929712462
2.1846522762634235	12.718849840255592
2.185831163457327	13.210862619808307
2.220623714253556	13.613418530351439
2.255748799054423	13.926517571884984
2.255970068843575	14.015974440894569
2.435343187173589	13.792332268370608
2.47240914208274	13.568690095846645
2.509916143602259	13.300319488817891
2.547992134751788	13.031948881789138
2.547742222888948	12.942492012779553
2.5472424726963405	12.763578274760384
2.585884712634639	12.49520766773163
2.585123901946731	12.226837060702875
2.5848703481302975	12.13738019169329
2.6649367198252545	11.95846645367412
2.7057624775554308	11.824281150159745
2.7049663968168707	11.555910543130992
2.788889517327686	11.421725239616613
2.8313364519693063	11.19808306709265
2.874711386608766	11.063897763578275
2.9184645312144477	10.840255591054312
2.9630289125374594	10.661341853035143
3.0545091262274338	10.39297124600639
3.054059749243192	10.258785942492011
3.3449347197682875	9.230031948881788
3.1006946429665376	10.079872204472842
3.246029801024168	9.856230031948881
3.2453930788300234	9.677316293929712
3.2951111943071494	9.543130990415335
3.5020613401114877	9.095846645367413
3.447698552067784	8.827476038338656
3.4996576261267096	8.46964856230032
3.6080595382385616	8.29073482428115
3.662614979821021	7.977635782747603
3.775508965117988	7.6645367412140555
3.891882718828082	7.3514376996805115
4.073902675725465	7.038338658146964
4.072304606697792	6.680511182108624
4.26255368274972	6.322683706070286
4.462128429352407	6.054313099041533
4.529597766895492	5.741214057507985
4.6703601576246125	5.651757188498401
4.814552319712819	5.383386581469647
4.887350460940839	5.070287539936102
5.116680337023466	4.891373801916931
5.43963494466012	4.712460063897762
5.522426259101111	4.488817891373801
5.870990736112753	4.309904153354632
6.052547858444415	4.08626198083067
6.434256983067591	3.8626198083067074
6.734851256908489	3.5047923322683694
7.27176846857922	3.460063897763577
7.970598769617529	3.1469648562300296
8.475349463592782	3.1469648562300296
8.60518840547276	3.012779552715653
9.877171346548483	2.7444089456868994
9.433550667216632	2.833865814696484
10.664078238599888	2.654952076677315
11.513112761822988	2.520766773162938
12.055117812022477	2.476038338658146
12.241598338236543	2.476038338658146
13.837733513610797	2.252396166134183
13.420670934728445	2.341853035143769
15.887064673045423	2.2076677316293907
16.633351518195692	2.0734824281150157
19.095769650509038	1.9840255591054294
19.99474491481308	1.9392971246006354
20.93501466425129	1.8498402555910527
20.93501466425129	1.8498402555910527
22.259665654857805	1.8051118210862622
25.163252361956864	1.6261980830670897
25.163252361956864	1.6261980830670897
25.94777399049993	1.6261980830670897
29.33100497427884	1.4025559105431284
29.33100497427884	1.4025559105431284
32.15765303275845	1.3130990415335457
"""
data_array_bump = np.genfromtxt(StringIO(data_str_bump))
R_bump = data_array_bump[:, 0]   # R/Rsun
xi_data = data_array_bump[:, 1]  # T_perp/T_parallel
# バンプモデルの定義（対数座標上のシグモイド）
def bump_model(x, A, D, k1, mu1, k2, mu2):
    logistic_rise = 1.0 / (1 + np.exp(-k1 * (np.log(x) - mu1)))
    logistic_fall = 1.0 / (1 + np.exp(-k2 * (np.log(x) - mu2)))
    return A + D * logistic_rise * (1 - logistic_fall)
initial_guess = [1.3, 13.0, 10.0, 0.3, 10.0, 1.0]
lower_bounds = [0.0,   0.0,  0.0, -5.0,  0.0, -5.0]
upper_bounds = [10.0,  50.0, 50.0,  5.0, 50.0,  5.0]
from scipy.optimize import curve_fit
popt, pcov = curve_fit(bump_model, R_bump, xi_data, p0=initial_guess, bounds=(lower_bounds, upper_bounds), maxfev=10000)
print("バンプモデルの最適化パラメータ:")
print("A =", popt[0])
print("D =", popt[1])
print("k1 =", popt[2])
print("mu1 =", popt[3])
print("k2 =", popt[4])
print("mu2 =", popt[5])
x_fit = np.linspace(np.min(R_bump), np.max(R_bump), 1000)
y_fit = bump_model(x_fit, *popt)
plt.figure(figsize=(8,5))
plt.scatter(R_bump, xi_data, color='red', label='Data')
plt.plot(x_fit, y_fit, label='Fitted Bump Model')
plt.xscale("log")
plt.xlabel("R/Rsun")
plt.ylabel("T_perp/T_parallel")
plt.title("Fitted Bump Model")
plt.legend()
plt.grid(True)
plt.show()

#############################################
# ③ β = 8π p / B^2 を計算してプロット
#############################################
# p = n * (k_B/m_p) * T, ここで T = Tpara * (1 + 2*xi)/3, Tpara は spline_Tpara の結果, xi はバンプモデルから得る
# CGS単位系: k_B = 1.38e-16 erg/K, m_p = 1.67e-24 g
k_b = 1.38e-16       # erg/K
m_p = 1.6726219e-24  # g

# n(R) と B(R) の定義（CGS単位系）
n_0 = 3e7    # Rsun=1 における数密度 [cm^-3]
B_0 = 10     # Rsun=1 における磁場 [G]
alpha = 2.59         # 数密度の減衰指数
beta_exponent = 1.66 # 磁場の減衰指数

R_target = np.linspace(1, 30, 60)  # R/Rsun
# Tpara_target は spline_Tpara により得た T_parallel の評価値
Tpara_target = spline_Tpara(np.log(R_target))
# xi_target はバンプモデルから評価（T_perp/T_parallel の値）
xi_target = bump_model(R_target, *popt)
# 温度 T の計算: T = Tpara * (1 + 2*xi)/3
T = Tpara_target * (1 + 2 * xi_target) / 3

# n(R) と B(R) の計算
n_R = n_0 * (R_target)**(-alpha)
B_R = B_0 * (R_target)**(-beta_exponent)

# 圧力 p の計算
p = n_R * k_b * T  # erg/cm^3

# β の計算: β = (4π p) / B^2
beta = 8 * np.pi * p / (B_R**2)

plt.figure(figsize=(10,6))
plt.plot(R_target, beta, 'b-', lw=2, label=r'$\beta=\frac{8\pi\,p}{B^2}$')
plt.xlabel(r'$R/R_\odot$', fontsize=18)
plt.ylabel(r'$\beta$', fontsize=18)
plt.title(r'$\beta$ vs $R/R_\odot$', fontsize=20)
plt.legend(fontsize=16)
plt.grid(True, which='both', ls='--')
plt.tight_layout()
plt.show()

#############################################
# ④ PDI 成長率の計算（β の値を利用）
#############################################
# ここでは、先に補間したβの値を使って、各 r_ratio での PDI 成長率（最大虚部）を求める
# （下記の式は例示用です）

# まず、R_target, beta の配列から補間関数を作成
B_perp_squared_0=0.001
beta_interp = UnivariateSpline(R_target, beta, s=0)
# r=1 のときの β の値
beta_at_1 = beta_interp(1)

def calculate_max_imaginary_part(r_ratio):
    # 補間関数からその r_ratio での β を取得
    beta_val = beta_interp(r_ratio)
    # B_perp_squared の r によるスケーリング（例として r^1 を用いる）
    B_perp_squared = B_perp_squared_0 * r_ratio**(0.92)

    # ここでの式は例示です。以下の Newton 法で用いる f, df は適当な例です。
    def newton_method(omega_hat, k_hat):
        f = ((omega_hat - k_hat) * (omega_hat**2 - beta_val * k_hat**2) *
             ((omega_hat + k_hat)**2 - 4) -
             (B_perp_squared * k_hat**2) * (omega_hat**3 + k_hat * omega_hat**2 - 3 * omega_hat + k_hat))
        df = (3 * B_perp_squared * k_hat**2 + beta_val * k_hat**4 + 4 * beta_val * k_hat**2 +
              4 * k_hat * omega_hat**3 + 5 * omega_hat**4 +
              3 * omega_hat**2 * (-B_perp_squared * k_hat**2 - beta_val * k_hat**2 - k_hat**2 - 4) +
              2 * omega_hat * (-B_perp_squared * k_hat**3 - beta_val * k_hat**3 - k_hat**3 + 4 * k_hat))
        if np.abs(df) < 1e-10:
            return omega_hat
        return omega_hat - f / df

    def root_of_func(initial_guesses, k_hat):
        roots = []
        for omega_hat in initial_guesses:
            for _ in range(300):
                omega_hat_new = newton_method(omega_hat, k_hat)
                if np.abs(omega_hat_new - omega_hat) < 1e-10:
                    if all(np.abs(omega_hat_new - root) > 1e-10 for root in roots):
                        roots.append(omega_hat_new)
                    break
                omega_hat = omega_hat_new
        return roots

    k_hat_values = np.linspace(0.85, 2, 1000)
    omega_hats_imag = []
    initial_guesses = [0.1+0.2j, 2+0.5j, -5+2j, -5-8j, 7-3j, 1+1j, 2+0.1j, 2-0.1j,
                       -2+2j, -2-2j, 3+3j, 3-3j, -4+0.5j, 4+0.5j, 0+3j, 0-3j,
                       1+2j, 1-2j, -1+0.5j, -1-0.5j, 0.5+1j, 0.5-1j, -0.5+0.2j,
                       -0.5-0.2j, 3.5+4j, 3.5-4j, -3.5+4j, -3.5-4j,
                       2.15+0.1j, 2.20+0.1j, 2.25+0.1j, 2.1+0.1j, 2.15+0.07j, 2.15+0.04j, 2.15+0.01j, -4-4j,
                       6+2j, 6-2j, -6+2j, -6-2j, 0.1+0.3j, 0.1-0.3j, -0.1+0.3j, -0.1-0.3j]

    for k_hat in k_hat_values:
        roots = root_of_func(initial_guesses, k_hat)
        for root in roots:
            omega_hats_imag.append(root.imag)
    return np.max(omega_hats_imag)

# r_ratios の範囲 (1 から 30) を 30 点生成
r_ratios = np.linspace(1, 30, 90)

with Pool() as pool:
    max_imaginary_parts = pool.map(calculate_max_imaginary_part, r_ratios)

plt.figure(figsize=(14, 8))
plt.plot(r_ratios, max_imaginary_parts, lw=2, color='tab:blue',
         label=f"$\\beta(r=1) = {beta_at_1:.3e}$")
plt.xlabel(r"$r/r_{0}$", fontsize=25)
plt.ylabel(r"$\gamma_{max} / \omega_{0}$", fontsize=25)
plt.title("Maximum Growth Rate (MHD) vs r", fontsize=25)
plt.tick_params(axis='both', labelsize=20)
plt.legend(fontsize=16)
plt.grid(True, which='both', ls='--')
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
from io import StringIO
from multiprocessing import Pool

# ======== 既存のデータ読み込みや補間、フィッティングのコード ===========
# （ここでは既に R_target, spline_Tpara, bump_model, popt, beta_para_interp, B_perp_squared_0, beta_interp などが定義されているものとします）

# ----- 例：β の補間関数（MHD計算用） -----
# ※ この beta_interp は、先のコードで作成した R_target, beta のスプライン補間関数です
# 例: beta_interp = UnivariateSpline(R_target, beta, s=0)
# --------------------------------------------------------------------------

# ======== MHD版 最大成長率計算用関数 ========
def calculate_max_imaginary_part_mhd(r_ratio):
    beta_val = beta_interp(r_ratio)  # 例：beta_interp = UnivariateSpline(R_target, beta, s=0)
    B_perp_squared = B_perp_squared_0 * r_ratio**(0.92)

    def newton_method(omega_hat, k_hat):
        f = ((omega_hat - k_hat) * (omega_hat**2 - beta_val * k_hat**2) *
             ((omega_hat + k_hat)**2 - 4) -
             (B_perp_squared * k_hat**2) * (omega_hat**3 + k_hat * omega_hat**2 - 3 * omega_hat + k_hat))
        df = (3 * B_perp_squared * k_hat**2 + beta_val * k_hat**4 + 4 * beta_val * k_hat**2 +
              4 * k_hat * omega_hat**3 + 5 * omega_hat**4 +
              3 * omega_hat**2 * (-B_perp_squared * k_hat**2 - beta_val * k_hat**2 - k_hat**2 - 4) +
              2 * omega_hat * (-B_perp_squared * k_hat**3 - beta_val * k_hat**3 - k_hat**3 + 4 * k_hat))
        if np.abs(df) < 1e-10:
            return omega_hat
        return omega_hat - f / df

    def root_of_func(initial_guesses, k_hat):
        roots = []
        for omega_hat in initial_guesses:
            for _ in range(300):
                omega_hat_new = newton_method(omega_hat, k_hat)
                if np.abs(omega_hat_new - omega_hat) < 1e-10:
                    if all(np.abs(omega_hat_new - root) > 1e-10 for root in roots):
                        roots.append(omega_hat_new)
                    break
                omega_hat = omega_hat_new
        return roots

    k_hat_values = np.linspace(0.85, 2, 1000)
    omega_hats_imag = []
    initial_guesses = [0.1+0.2j, 2+0.5j, -5+2j, -5-8j, 7-3j, 1+1j, 2+0.1j, 2-0.1j,
                       -2+2j, -2-2j, 3+3j, 3-3j, -4+0.5j, 4+0.5j, 0+3j, 0-3j,
                       1+2j, 1-2j, -1+0.5j, -1-0.5j, 0.5+1j, 0.5-1j, -0.5+0.2j,
                       -0.5-0.2j, 3.5+4j, 3.5-4j, -3.5+4j, -3.5-4j,
                       2.15+0.1j, 2.20+0.1j, 2.25+0.1j, 2.1+0.1j, 2.15+0.07j, 2.15+0.04j, 2.15+0.01j, -4-4j,
                       6+2j, 6-2j, -6+2j, -6-2j, 0.1+0.3j, 0.1-0.3j, -0.1+0.3j, -0.1-0.3j]
    for k_hat in k_hat_values:
        roots = root_of_func(initial_guesses, k_hat)
        for root in roots:
            omega_hats_imag.append(root.imag)
    return np.max(omega_hats_imag)

# ======== PDI版 最大成長率計算用関数 ========
def calculate_max_imaginary_part_pdi(r_ratio):
    beta_parallel = beta_para_interp(r_ratio)
    xi = bump_model(r_ratio, *popt)
    B_perp_squared = B_perp_squared_0 * r_ratio**(0.92)
    tilde_beta = beta_parallel * (3.0/2.0) / (1.0 + B_perp_squared + beta_parallel/2.0 * (xi - 1.0))

    def newton_method(omega_hat, k_hat):
        f = ((omega_hat**2 - tilde_beta * k_hat**2 * (1.0 + B_perp_squared * xi / 3.0)) *
             ((omega_hat - k_hat) * ((omega_hat + k_hat)**2 - 4.0) +
              tilde_beta * B_perp_squared * (xi - 4.0) / (3.0*(1.0+B_perp_squared)) *
             ((k_hat**2+1.0)*omega_hat + k_hat*(k_hat**2-3.0))) -
             (B_perp_squared * k_hat**2 * (1.0 - tilde_beta*(3.0-xi-B_perp_squared) / (3.0*(1.0+B_perp_squared))) *
              ((omega_hat**3+omega_hat**2*k_hat-3.0*omega_hat+k_hat) -
               tilde_beta*(3.0-xi)/3.0 * ((k_hat**2+1.0)*omega_hat+k_hat*(k_hat**2-3.0)))))
        df = (2*omega_hat * (((omega_hat-k_hat)*((omega_hat+k_hat)**2-4.0)) +
              tilde_beta*B_perp_squared*(xi-4.0)/(3.0*(1.0+B_perp_squared)) *
              ((k_hat**2+1.0)*omega_hat+k_hat*(k_hat**2-3.0))) +
              (omega_hat**2 - tilde_beta*k_hat**2*(1.0+B_perp_squared*xi/3.0)) *
              (((omega_hat+k_hat)**2-4.0) + (omega_hat-k_hat)*2*(omega_hat+k_hat)) +
              tilde_beta*B_perp_squared*(xi-4.0)/(3.0*(1.0+B_perp_squared))*(k_hat**2+1.0) -
              B_perp_squared*k_hat**2*(1.0 - tilde_beta*(3.0-xi-B_perp_squared) / (3.0*(1.0+B_perp_squared))) *
              (3*omega_hat**2+2*omega_hat*k_hat-3.0) -
              (B_perp_squared*k_hat**2 * (-tilde_beta*(3.0-xi)/3.0))*(k_hat**2+1.0)
             )
        if np.abs(df) < 1e-10:
            return omega_hat
        return omega_hat - f/df

    def root_of_func(initial_guesses, k_hat):
        roots = []
        for omega_hat in initial_guesses:
            for _ in range(1000):
                omega_hat_new = newton_method(omega_hat, k_hat)
                if np.abs(omega_hat_new - omega_hat) < 1e-10:
                    if all(np.abs(omega_hat_new - root) > 1e-10 for root in roots):
                        roots.append(omega_hat_new)
                    break
                omega_hat = omega_hat_new
        return roots

    k_hat_values = np.linspace(0.85, 2, 1000)
    omega_hats_imag = []
    initial_guesses = [0.1+0.2j, 2+0.5j, -5+2j, -5-8j, 7-3j, 1+1j, 2+0.1j, 2-0.1j,
                       -2+2j, -2-2j, 3+3j, 3-3j, -4+0.5j, 4+0.5j, 0+3j, 0-3j,
                       1+2j, 1-2j, -1+0.5j, -1-2j, 0.5+1j, 0.5-1j, -0.5+0.2j,
                       -0.5-0.2j, 3.5+4j, 3.5-4j, -3.5+4j, -3.5-4j,
                       2.15+0.1j, 2.20+0.1j, 2.25+0.1j, 2.1+0.1j, 2.15+0.07j, 2.15+0.04j, 2.15+0.01j, -4-4j,
                       6+2j, 6-2j, -6+2j, -6-2j, 0.1+0.3j, 0.1-0.3j, -0.1+0.3j, -0.1-0.3j]
    for k_hat in k_hat_values:
        roots = root_of_func(initial_guesses, k_hat)
        for root in roots:
            omega_hats_imag.append(root.imag)
    return np.max(omega_hats_imag)

# ======== 両者の計算と同一Figureでのプロット ============
# 共通の r_ratio の範囲（例：1～30）
r_ratios = np.linspace(1, 30, 90)

# MHD版最大成長率の計算
with Pool() as pool:
    max_imag_mhd = pool.map(calculate_max_imaginary_part_mhd, r_ratios)

# PDI版の場合は Firehose 条件により計算可能な範囲を制限する（例）
def calculate_firehose_condition(r):
    beta_parallel = beta_para_interp(r)
    xi = bump_model(r, *popt)
    B_perp_squared = B_perp_squared_0 * r**(0.92)
    return 1 + 0.5 * beta_parallel * (xi - 1) / (1 + B_perp_squared)

firehose_values = np.array([calculate_firehose_condition(r) for r in r_ratios])
cutoff_index = np.argmax(firehose_values < 0) if np.any(firehose_values < 0) else len(r_ratios)
r_ratios_pdi = r_ratios[:cutoff_index]

with Pool() as pool:
    max_imag_pdi = pool.map(calculate_max_imaginary_part_pdi, r_ratios_pdi)

# プロット
plt.rcParams["font.size"] = 20
plt.figure(figsize=(14, 8))
plt.plot(r_ratios, max_imag_mhd, lw=2, color='tab:blue', label=r"$\gamma_{max}/\omega_{0} (isotropic)$")
plt.plot(r_ratios_pdi, max_imag_pdi, lw=2, color='tab:red', label=r"$\gamma_{max}/\omega_{0}$ (CGL)")
plt.xlabel(r"$R/R_{0}$", fontsize=25)
plt.ylabel(r"$\gamma_{max}/\omega_{0}$", fontsize=25)
plt.title("$\gamma_{max}/\omega_{0}$ vs r", fontsize=25)
plt.legend(fontsize=18)
plt.grid(True)
plt.tight_layout()
plt.show()

# ▼ 追加セル ──────────────────────────────────────────────
# ---------------------------------------------------------
# 1) 物理パラメータを R 方向に並べた配列を準備
# ---------------------------------------------------------
beta_iso_vals       = beta_interp(r_ratios)              # 等方 MHD の β
beta_parallel_vals  = beta_para_interp(r_ratios)         # CGL β||
xi_vals             = bump_model(r_ratios, *popt)        # CGL ξ
B_perp_vals         = B_perp_squared_0 * r_ratios**(0.92)     # B⊥² (r¹ スケーリング)

# ---------------------------------------------------------
# 2) 連結プロット作成
# ---------------------------------------------------------
fig, axes = plt.subplots(
    nrows=2, ncols=1, sharex=True, figsize=(14, 10),
    gridspec_kw={'hspace': 0.04, 'height_ratios': [1.5, 1]}
)

# ── (i) 上段：成長率比較 ──────────────────────────────
ax_top = axes[0]
ax_top.plot(r_ratios,      max_imag_mhd, lw=2, color='tab:blue',
            label=r'$\gamma_{\max}/\omega_{0}\;$(isotropic)')
ax_top.plot(r_ratios_pdi,  max_imag_pdi, lw=2, color='tab:red',
            label=r'$\gamma_{\max}/\omega_{0}\;$(CGL)')

ax_top.set_ylabel(r'$\gamma_{\max}/\omega_{0}$')
ax_top.grid(True, alpha=0.3)
ax_top.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

# ── (ii) 下段：β, β∥, ξ, 𝐵⊥² ─────────────────────────
ax_bot = axes[1]

ax_bot.plot(r_ratios, beta_iso_vals,      lw=2, color='tab:brown',
            label=r'$\beta$ (isotropic)')
ax_bot.plot(r_ratios, beta_parallel_vals, lw=2, color='tab:purple',
            label=r'$\beta_{\parallel}$ (CGL)')
ax_bot.plot(r_ratios, xi_vals,            lw=2, color='tab:orange',
            label=r'$\xi=T_{\perp}/T_{\parallel}$')
ax_bot.plot(r_ratios, B_perp_vals,        lw=2, color='tab:cyan',
            label=r'$\hat{B}_{\perp}^{2}$')

ax_bot.set_yscale('log')
ax_bot.set_xlabel(r'$R/R_{0}$')
ax_bot.set_ylabel(r'$\beta,\;\beta_{\parallel},\;\xi,\;\hat{B}_{\perp}^{2}$')
ax_bot.set_ylim(1e-5, 1e2)
ax_bot.set_yticks([1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2])
from matplotlib.ticker import LogLocator
# ───── log‐grid 追加ここから ─────
ax_bot.yaxis.set_minor_locator(LogLocator(base=10.0,          # 10 のべき
                                       subs=np.arange(1, 10)*0.1,  # 1–9 の副目盛り
                                       numticks=100))
ax_bot.grid(True,  which='major', alpha=0.6)  # メジャー
ax_bot.grid(True,  which='minor', axis='y', alpha=0.3)  # マイナー
# ───── ここまで ─────
ax_bot.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

# ── レイアウト調整 ──────────────────────────────────
plt.subplots_adjust(right=0.8)
plt.tight_layout()
plt.show()
# ▲ 追加セル ──────────────────────────────────────────────

# ▼▼▼ ここから追記 ─────────────────────────────────────────
# -----------------------------------------------------------
# 6. データセット（90 × 7）を作成して CSV に保存
#     0: r_ratio
#     1: γ_max/ω0  (CGL - PDI)           … Firehoseが負になる r では NaN
#     2: γ_max/ω0  (isotropic-MHD)
#     3: β         (isotropic-MHD)
#     4: β_parallel(CGL - PDI)
#     5: ξ         (=T_perp/T_parallel)
#     6: B_perp²   (≡ B_perp_squared_0 · r^0.92)
# -----------------------------------------------------------

# ---- ① CGL 成長率を全 r 90 点に並べる（Firehoseカット以降は NaN） ----
gamma_CGL_full = np.full_like(r_ratios, np.nan, dtype=float)
gamma_CGL_full[:cutoff_index] = max_imag_pdi   # PDI 計算で得た値を先頭から代入

# ---- ② 列方向にスタックしてデータセット完成 ----------------------------
dataset = np.column_stack([
    r_ratios,           # 0
    gamma_CGL_full,     # 1
    max_imag_mhd,       # 2
    beta_iso_vals,      # 3
    beta_parallel_vals, # 4
    xi_vals,            # 5
    B_perp_vals         # 6
])

# ---- ③ CSV 出力 ---------------------------------------------------------
header = ("R_over_R0, gamma_CGL, gamma_isotropic, "
          "beta_isotropic, beta_parallel, xi, B_perp_squared")

np.savetxt(
    "Fig4a_dataset_full.csv",
    dataset,
    delimiter=",",
    header=header,
    comments=""
)

print(">> Fig4a_dataset_full.csv を保存しました")
print("\n--- preview ---")
print(dataset[:5])   # 先頭5行を確認
# ▲▲▲ ここまで追記 ─────────────────────────────────────────


#シナリオ４＆５　beta=0.01

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
from io import StringIO
from multiprocessing import Pool
import sympy as sp

#############################
# ① βₚₐᵣₐ の計算（Tₚₐᵣₐ データから）
#############################

# --- Tₚₐᵣₐ のデータ定義 ---
data_str_Tpara = """
0.9363634313172465	630950.7344480193
0.9518215986713207	735640.22544596407
0.952918054833259	874310.24580220731
0.9679063371940468	908510.75756516862
0.998210703186863	926110.87281287937
1.0136520170010803	926110.87281287937
1.110736672952074	841390.51416451947
1.2356312656535287	735640.22544596407
1.3529376348534548	595660.214352901094
1.5242477252204125	348070.00588428406
1.5691579687708173	271220.725793320296
1.6153914420970539	211340.89039836648
1.6640511310707182	181270.308811846968
1.7680722177343189	161550.98098439873
1.820865124259488	133350.21432163324
2.0274233521107727	133350.21432163324
2.294386965053555	152520.229565390198
2.4434305526939717	192010.41938638801
2.6014903677573367	232630.05067153624
2.856128729537577	281830.82931264455
3.0404960563039145	334960.54391578279
3.3897421734448465	405810.98942243799
3.665782272509721	510890.69774506924
3.9627803425543924	607200.21956909884
4.6977390496362625	735640.22544596407
5.484172555736265	891250.0938133744
6.1125471512033736	1039120.2303835169
7.241580008400507	1143750.58630495366
8.447383717827934	1234990.91726875826
10.966184681342897	1234990.91726875826
12.58764378600493	1188500.22274370166
15.127869790847797	1122010.84543019629
18.180721372437116	1059250.37251772876
21.18632136291081	980990.47127268763
25.471560675002788	980990.47127268763
30.15705228022852	980990.47127268763
1.3518996152492362	530880.44442309879
1.4147075349881577	482310.78482239309
1.4580679970023382	446680.35921509626
1.4796761259208662	405810.98942243799
1.523078270709203	310210.776647994982
1.5679540575987572	241730.154808041014
1.2539429491545022	668340.39175686135
1.766941707999398	146770.992676220705
2.1566406048857756	141250.375446227554
2.3680392085234327	174440.826989992558
2.4828154797631	211340.89039836648
2.6853444456585054	271220.725793320296
2.993030283874072	316220.776601683792
3.235108784158188	368690.450645195735
3.4443804148047565	446680.35921509626
4.152726538879247	681290.20690579608
4.922913696391965	825400.41852680173
9.405651624378091	1234990.91726875826
6.603554671772791	1122010.84543019629
5.745572820257177	962350.06263980868
13.59182511235687	1188500.22274370166
16.33260863538179	1100690.4171252208
22.87939016505923	1000000
28.364672776598606	1000000
19.0399878392478	1079770.51623277094
19.3295713029763	1039120.2303835169
7.821278990952227	1188500.22274370166
1.1449269489307676	794320.82347242805
32.070897303318624	1000000
34.62492878921517	980990.47127268763
37.965479460384884	980990.47127268763
42.26686142656028	962350.06263980868
47.04354057259765	908510.75756516862
"""

# --- 文字列からデータを読み込み ---
data_Tpara = np.genfromtxt(data_str_Tpara.strip().splitlines())
R_Tpara = data_Tpara[:, 0]         # R/Rsun
Tpara = data_Tpara[:, 1]           # T_para

# --- R の昇順にソート ---
idx = np.argsort(R_Tpara)
R_Tpara = R_Tpara[idx]
Tpara = Tpara[idx]

# --- R 軸の対数変換 ---
log_R_Tpara = np.log(R_Tpara)

# --- スプライン補間（smoothing_parameter は適宜調整） ---
smoothing_parameter = 1e11
spline_Tpara = UnivariateSpline(log_R_Tpara, Tpara, s=smoothing_parameter)

# --- R/Rsun = 1 から 30 の範囲を 60 分割 ---
R_target = np.linspace(1, 30, 60)

# --- 対数変換してスプライン関数で T_para を評価 ---
T_target = spline_Tpara(np.log(R_target))

# --- 定数および初期値 ---
k_b = 1.38e-16        # ボルツマン定数 [J/K]
n_0 = 3e8             # r=1 における数密度の初期値（任意の値）
B_0 = 9.7              # r=1 における磁場の初期値（任意の値）

# --- Rsun=1 における T_para ---
Tpara_0 = spline_Tpara(np.log(1))
print("Rsun=1 のときの T_para =", Tpara_0)

# --- R に依存する n(R) と B(R) の定義 ---
n_R = n_0 * (R_target)**(-2.59)
B_R = B_0 * (R_target)**(-1.66)

# --- β_parallel の計算 ---
beta_para = 8 * np.pi * n_R * k_b * T_target / (B_R**2)

# --- β_parallel のプロット（参考） ---
plt.figure(figsize=(10,6))
plt.plot(R_target, beta_para, 'b-', lw=2, label=r'$\beta_{\parallel}$')
plt.xlabel(r'$R/R_\odot$')
plt.ylabel(r'$\beta_{\parallel}$')
plt.title(r'$\beta_{\parallel}$ vs $R/R_\odot$')
plt.legend()
plt.grid(True)
plt.show()

# --- 補間関数を作成（PDI 用に β_parallel を任意の r で評価） ---
beta_para_interp = UnivariateSpline(R_target, beta_para, s=0)

#############################
# ② T_perp/T_para（=xi）のバンプモデルフィッティング
#############################

# --- T_perp/T_para のデータ定義 ---
data_str_bump = """
1.0009813481265506	0.910543130990412
1.0810468494787815	1.0894568690095827
1.149562217230484	1.1341853035143732
1.2413904876306563	1.2236421725239612
1.2998954188768872	1.2236421725239612
1.3611576025865708	1.2236421725239612
1.4036636255995907	1.2683706070287517
1.4698162542259607	1.2683706070287517
1.5396905343022556	1.6261980830670897
1.6373544520272827	1.7156549520766742
1.6630904432580889	1.9392971246006354
1.715782341439801	2.3865814696485597
1.7428365372600465	2.654952076677315
1.7436059778765893	3.057507987220447
1.7988488596018384	3.5047923322683694
1.7992900215245589	3.7284345047923306
1.8003492516192892	4.265175718849839
1.8288265752326527	4.578274760383385
1.8578454563701159	4.936102236421723
1.858665671991507	5.338658146964855
1.917459963137651	5.741214057507985
1.9184005802972979	6.1884984025559095
1.9191534062389042	6.546325878594249
1.9494143261746528	6.814696485623003
1.9804437570384015	7.217252396166133
1.9811237668749409	7.5303514376996805
1.9821928255364416	8.022364217252395
2.014040305942652	8.559105431309902
2.01483066318364	8.916932907348244
2.0463994732015016	9.095846645367413
2.047202528948116	9.45367412140575
2.0483072451864515	9.945686900958465
2.0810128425639136	10.39297124600639
2.0816252916659104	10.661341853035143
2.082237921013744	10.929712460063897
2.0828507306604624	11.19808306709265
2.1820823652270755	11.645367412140576
2.183045727403937	12.047923322683706
2.1836882051282536	12.316293929712462
2.1846522762634235	12.718849840255592
2.185831163457327	13.210862619808307
2.220623714253556	13.613418530351439
2.255748799054423	13.926517571884984
2.255970068843575	14.015974440894569
2.435343187173589	13.792332268370608
2.47240914208274	13.568690095846645
2.509916143602259	13.300319488817891
2.547992134751788	13.031948881789138
2.547742222888948	12.942492012779553
2.5472424726963405	12.763578274760384
2.585884712634639	12.49520766773163
2.585123901946731	12.226837060702875
2.5848703481302975	12.13738019169329
2.6649367198252545	11.95846645367412
2.7057624775554308	11.824281150159745
2.7049663968168707	11.555910543130992
2.788889517327686	11.421725239616613
2.8313364519693063	11.19808306709265
2.874711386608766	11.063897763578275
2.9184645312144477	10.840255591054312
2.9630289125374594	10.661341853035143
3.0545091262274338	10.39297124600639
3.054059749243192	10.258785942492011
3.3449347197682875	9.230031948881788
3.1006946429665376	10.079872204472842
3.246029801024168	9.856230031948881
3.2453930788300234	9.677316293929712
3.2951111943071494	9.543130990415335
3.5020613401114877	9.095846645367413
3.447698552067784	8.827476038338656
3.4996576261267096	8.46964856230032
3.6080595382385616	8.29073482428115
3.662614979821021	7.977635782747603
3.775508965117988	7.6645367412140555
3.891882718828082	7.3514376996805115
4.073902675725465	7.038338658146964
4.072304606697792	6.680511182108624
4.26255368274972	6.322683706070286
4.462128429352407	6.054313099041533
4.529597766895492	5.741214057507985
4.6703601576246125	5.651757188498401
4.814552319712819	5.383386581469647
4.887350460940839	5.070287539936102
5.116680337023466	4.891373801916931
5.43963494466012	4.712460063897762
5.522426259101111	4.488817891373801
5.870990736112753	4.309904153354632
6.052547858444415	4.08626198083067
6.434256983067591	3.8626198083067074
6.734851256908489	3.5047923322683694
7.27176846857922	3.460063897763577
7.970598769617529	3.1469648562300296
8.475349463592782	3.1469648562300296
8.60518840547276	3.012779552715653
9.877171346548483	2.7444089456868994
9.433550667216632	2.833865814696484
10.664078238599888	2.654952076677315
11.513112761822988	2.520766773162938
12.055117812022477	2.476038338658146
12.241598338236543	2.476038338658146
13.837733513610797	2.252396166134183
13.420670934728445	2.341853035143769
15.887064673045423	2.2076677316293907
16.633351518195692	2.0734824281150157
19.095769650509038	1.9840255591054294
19.99474491481308	1.9392971246006354
20.93501466425129	1.8498402555910527
20.93501466425129	1.8498402555910527
22.259665654857805	1.8051118210862622
25.163252361956864	1.6261980830670897
25.163252361956864	1.6261980830670897
25.94777399049993	1.6261980830670897
29.33100497427884	1.4025559105431284
29.33100497427884	1.4025559105431284
32.15765303275845	1.3130990415335457
"""

# 文字列から数値データを読み込む
data_array_bump = np.genfromtxt(StringIO(data_str_bump))
R_bump = data_array_bump[:, 0]  # R/Rsun
xi_data = data_array_bump[:, 1] # T_perp/T_para

# バンプモデルの定義（対数座標上のシグモイド）
def bump_model(x, A, D, k1, mu1, k2, mu2):
    logistic_rise = 1.0 / (1 + np.exp(-k1 * (np.log(x) - mu1)))
    logistic_fall = 1.0 / (1 + np.exp(-k2 * (np.log(x) - mu2)))
    return A + D * logistic_rise * (1 - logistic_fall)

# 初期推定値とパラメータ境界
initial_guess = [1.3, 13.0, 10.0, 0.3, 10.0, 1.0]
lower_bounds = [0.0,   0.0,  0.0, -5.0,  0.0, -5.0]
upper_bounds = [10.0,  50.0, 50.0,  5.0, 50.0,  5.0]

# フィッティング実行
popt, pcov = curve_fit(
    bump_model, R_bump, xi_data,
    p0=initial_guess,
    bounds=(lower_bounds, upper_bounds),
    maxfev=10000
)

print("バンプモデルの最適化パラメータ:")
print("A =", popt[0])
print("D =", popt[1])
print("k1 =", popt[2])
print("mu1 =", popt[3])
print("k2 =", popt[4])
print("mu2 =", popt[5])

# フィッティング結果のプロット（参考）
x_fit = np.linspace(np.min(R_bump), np.max(R_bump), 1000)
y_fit = bump_model(x_fit, *popt)
plt.figure(figsize=(8,5))
plt.scatter(R_bump, xi_data, color='red', label='Data')
plt.plot(x_fit, y_fit, label='Fitted Bump Model')
plt.xscale("log")
plt.xlabel("R/Rsun")
plt.ylabel("T_perp/T_para")
plt.title("Fitted Bump Model")
plt.legend()
plt.grid(True)
plt.show()

#############################
# ③ PDI 成長率と Firehose 条件の計算・プロット
#############################

# 固定定数（r=1 における B_perp² の値）
B_perp_squared_0 = 0.001

# ※ ここでは T_perp/T_para = xi をバンプモデルから得るので，
#    固定値 xi0 は不要となります。

def calculate_max_imaginary_part(r_ratio):
    # beta_parallel は補間関数から取得
    beta_parallel = beta_para_interp(r_ratio)
    # xi を固定のスケーリングではなく，バンプモデルから求める．
    xi = bump_model(r_ratio, *popt)
    # ※ B_perp_squared の r によるスケーリングはここでは例として r^1 を用いる（必要に応じて調整）
    B_perp_squared = B_perp_squared_0 * r_ratio**(0.92)
    tilde_beta = beta_parallel * (3.0/2.0) / (1.0 + B_perp_squared + beta_parallel/2.0 * (xi - 1.0))

    def newton_method(omega_hat, k_hat):
        f = ((omega_hat**2 - tilde_beta * k_hat**2 * (1.0 + B_perp_squared * xi / 3.0)) *
             ((omega_hat - k_hat) * ((omega_hat + k_hat)**2 - 4.0) +
              tilde_beta * B_perp_squared * (xi - 4.0) / (3.0*(1.0+B_perp_squared)) *
             ((k_hat**2+1.0)*omega_hat + k_hat*(k_hat**2-3.0))) -
             (B_perp_squared * k_hat**2 * (1.0 - tilde_beta*(3.0-xi-B_perp_squared) / (3.0*(1.0+B_perp_squared))) *
              ((omega_hat**3+omega_hat**2*k_hat-3.0*omega_hat+k_hat) -
               tilde_beta*(3.0-xi)/3.0 * ((k_hat**2+1.0)*omega_hat+k_hat*(k_hat**2-3.0)))))
        df = (2*omega_hat * (((omega_hat-k_hat)*((omega_hat+k_hat)**2-4.0)) +
              tilde_beta*B_perp_squared*(xi-4.0)/(3.0*(1.0+B_perp_squared)) *
              ((k_hat**2+1.0)*omega_hat+k_hat*(k_hat**2-3.0))) +
              (omega_hat**2 - tilde_beta*k_hat**2*(1.0+B_perp_squared*xi/3.0)) *
              (((omega_hat+k_hat)**2-4.0) + (omega_hat-k_hat)*2*(omega_hat+k_hat)) +
              tilde_beta*B_perp_squared*(xi-4.0)/(3.0*(1.0+B_perp_squared))*(k_hat**2+1.0) -
              B_perp_squared*k_hat**2*(1.0 - tilde_beta*(3.0-xi-B_perp_squared) / (3.0*(1.0+B_perp_squared))) *
              (3*omega_hat**2+2*omega_hat*k_hat-3.0) -
              (B_perp_squared*k_hat**2 * (-tilde_beta*(3.0-xi)/3.0))*(k_hat**2+1.0)
             )
        if np.abs(df) < 1e-10:
            return omega_hat
        return omega_hat - f/df

    def root_of_func(initial_guesses, k_hat):
        roots = []
        for omega_hat in initial_guesses:
            for _ in range(1000):
                omega_hat_new = newton_method(omega_hat, k_hat)
                if np.abs(omega_hat_new - omega_hat) < 1e-10:
                    if all(np.abs(omega_hat_new - root) > 1e-10 for root in roots):
                        roots.append(omega_hat_new)
                    break
                omega_hat = omega_hat_new
        return roots

    # k_hat の範囲を決め，各 k_hat での根の虚部の最大値を求める
    k_hat_values = np.linspace(0.95, 2, 100)
    omega_hats_imag = []
    initial_guesses = [0.1+0.2j, 2+0.5j, -5+2j, -5-8j, 7-3j, 1+1j, 2+0.1j, 2-0.1j,
                       -2+2j, -2-2j, 3+3j, 3-3j, -4+0.5j, 4+0.5j, 0+3j, 0-3j,
                       1+2j, 1-2j, -1+0.5j, -1-0.5j, 0.5+1j, 0.5-1j, -0.5+0.2j,
                       -0.5-0.2j, 3.5+4j, 3.5-4j, -3.5+4j, -3.5-4j,
                       2.15+0.1j, 2.20+0.1j, 2.25+0.1j, 2.1+0.1j, 2.15+0.07j, 2.15+0.04j, 2.15+0.01j,
                       -4-4j, 6+2j, 6-2j, -6+2j, -6-2j, 0.1+0.3j, 0.1-0.3j, -0.1+0.3j, -0.1-0.3j]
    for k_hat in k_hat_values:
        roots = root_of_func(initial_guesses, k_hat)
        for root in roots:
            omega_hats_imag.append(root.imag)
    return np.max(omega_hats_imag)

def calculate_firehose_condition(r_ratios):
    beta_parallel_values = beta_para_interp(r_ratios)
    # xi を各 r で，バンプモデルから求める
    xi_values = bump_model(r_ratios, *popt)
    B_perp_squared_values = B_perp_squared_0 * r_ratios**(0.92)
    condition = 1 + 0.5 * beta_parallel_values * (xi_values - 1) / (1 + B_perp_squared_values)
    return condition

# r_ratio の範囲（1～30）
r_ratios = np.linspace(1, 30, 90)

# Firehose 条件の計算
firehose_condition = calculate_firehose_condition(r_ratios)
cutoff_index = np.argmax(firehose_condition < 0) if np.any(firehose_condition < 0) else len(r_ratios)
r_ratios_cutoff = r_ratios[:cutoff_index]

# 並列計算で各 r_ratio に対する最大虚部を求める
with Pool() as pool:
    max_imaginary_parts = pool.map(calculate_max_imaginary_part, r_ratios_cutoff)

# プロット①：PDI 成長率（最大虚部） vs r_ratio
plt.figure(figsize=(14,8))
plt.plot(r_ratios_cutoff, max_imaginary_parts, lw=2, color='tab:blue',
         label=f"$\\beta_{{||}}(r=1) = {beta_para_interp(1):.3e}$, $\\xi(r=1) = {bump_model(1, *popt):.3f}$")
plt.xlabel(r"$r/r_{0}$", fontsize=20)
plt.ylabel(r"$\gamma/\omega_{0}$", fontsize=20)
plt.title("Maximum Growth Rate (PDI) vs r", fontsize=22)
plt.legend(fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.show()

# プロット②：Firehose 条件 vs r_ratio
plt.figure(figsize=(14,8))
plt.plot(r_ratios, firehose_condition, lw=2, color='tab:green',
         label=f"$\\xi(r=1) = {bump_model(1, *popt):.3f}$, $\\beta_{{||}}(r=1) = {beta_para_interp(1):.3e}$")
plt.axhline(0, color='black', lw=1, ls='--')
plt.xlabel(r"$r/r_{0}$", fontsize=20)
plt.ylabel("Discriminant Value", fontsize=20)
plt.title("Firehose Condition vs r", fontsize=22)
plt.xlim(1, 30)
plt.ylim(-1, 1.5)
plt.legend(fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
from io import StringIO
from multiprocessing import Pool
import sympy as sp

# フォントサイズの設定
plt.rcParams.update({'font.size': 16})

#############################################
# ① T_parallel のスプライン補間によるフィッティング
#############################################
data_str_Tpara = """
0.9363634313172465	630950.7344480193
0.9518215986713207	735640.22544596407
0.952918054833259	874310.24580220731
0.9679063371940468	908510.75756516862
0.998210703186863	926110.87281287937
1.0136520170010803	926110.87281287937
1.110736672952074	841390.51416451947
1.2356312656535287	735640.22544596407
1.3529376348534548	595660.214352901094
1.5242477252204125	348070.00588428406
1.5691579687708173	271220.725793320296
1.6153914420970539	211340.89039836648
1.6640511310707182	181270.308811846968
1.7680722177343189	161550.98098439873
1.820865124259488	133350.21432163324
2.0274233521107727	133350.21432163324
2.294386965053555	152520.229565390198
2.4434305526939717	192010.41938638801
2.6014903677573367	232630.05067153624
2.856128729537577	281830.82931264455
3.0404960563039145	334960.54391578279
3.3897421734448465	405810.98942243799
3.665782272509721	510890.69774506924
3.9627803425543924	607200.21956909884
4.6977390496362625	735640.22544596407
5.484172555736265	891250.0938133744
6.1125471512033736	1039120.2303835169
7.241580008400507	1143750.58630495366
8.447383717827934	1234990.91726875826
10.966184681342897	1234990.91726875826
12.58764378600493	1188500.22274370166
15.127869790847797	1122010.84543019629
18.180721372437116	1059250.37251772876
21.18632136291081	980990.47127268763
25.471560675002788	980990.47127268763
30.15705228022852	980990.47127268763
1.3518996152492362	530880.44442309879
1.4147075349881577	482310.78482239309
1.4580679970023382	446680.35921509626
1.4796761259208662	405810.98942243799
1.523078270709203	310210.776647994982
1.5679540575987572	241730.154808041014
1.2539429491545022	668340.39175686135
1.766941707999398	146770.992676220705
2.1566406048857756	141250.375446227554
2.3680392085234327	174440.826989992558
2.4828154797631	211340.89039836648
2.6853444456585054	271220.725793320296
2.993030283874072	316220.776601683792
3.235108784158188	368690.450645195735
3.4443804148047565	446680.35921509626
4.152726538879247	681290.20690579608
4.922913696391965	825400.41852680173
9.405651624378091	1234990.91726875826
6.603554671772791	1122010.84543019629
5.745572820257177	962350.06263980868
13.59182511235687	1188500.22274370166
16.33260863538179	1100690.4171252208
22.87939016505923	1000000
28.364672776598606	1000000
19.0399878392478	1079770.51623277094
19.3295713029763	1039120.2303835169
7.821278990952227	1188500.22274370166
1.1449269489307676	794320.82347242805
32.070897303318624	1000000
34.62492878921517	980990.47127268763
37.965479460384884	980990.47127268763
42.26686142656028	962350.06263980868
47.04354057259765	908510.75756516862
"""
# データ読み込み
data_Tpara = np.genfromtxt(StringIO(data_str_Tpara))
R_Tpara = data_Tpara[:, 0]         # R/Rsun
Tpara = data_Tpara[:, 1]           # T_parallel [K]
idx = np.argsort(R_Tpara)
R_Tpara = R_Tpara[idx]
Tpara = Tpara[idx]
log_R_Tpara = np.log(R_Tpara)
smoothing_parameter = 1e11
spline_Tpara = UnivariateSpline(log_R_Tpara, Tpara, s=smoothing_parameter)
R_target = np.linspace(1, 30, 60)
Tpara_target = spline_Tpara(np.log(R_target))
print("R/Rsun (60点):")
print(R_target)
print("\n対応する T_parallel:")
print(Tpara_target)
plt.figure(figsize=(10, 6))
plt.plot(R_Tpara, Tpara, 'ro', label='data')
plt.plot(R_target, Tpara_target, 'gs', label='evaluated points')
plt.plot(np.exp(np.linspace(log_R_Tpara.min(), log_R_Tpara.max(), 2000)),
         spline_Tpara(np.linspace(log_R_Tpara.min(), log_R_Tpara.max(), 2000)),
         'b-', lw=2, label='smooth fitting spline')
plt.xlabel('R/Rsun (log scale)')
plt.ylabel('T_parallel')
plt.ylim(0,2000000)
plt.title('Smooth fitting spline for R/Rsun and T_parallel\n(評価点: R/Rsun=[1, 30] 60点)')
plt.legend()
plt.grid(True, which='both', ls='--')
plt.show()

#############################################
# ② T_perp/T_parallel のバンプモデルによるフィッティング
#############################################
data_str_bump = """
1.0009813481265506	0.910543130990412
1.0810468494787815	1.0894568690095827
1.149562217230484	1.1341853035143732
1.2413904876306563	1.2236421725239612
1.2998954188768872	1.2236421725239612
1.3611576025865708	1.2236421725239612
1.4036636255995907	1.2683706070287517
1.4698162542259607	1.2683706070287517
1.5396905343022556	1.6261980830670897
1.6373544520272827	1.7156549520766742
1.6630904432580889	1.9392971246006354
1.715782341439801	2.3865814696485597
1.7428365372600465	2.654952076677315
1.7436059778765893	3.057507987220447
1.7988488596018384	3.5047923322683694
1.7992900215245589	3.7284345047923306
1.8003492516192892	4.265175718849839
1.8288265752326527	4.578274760383385
1.8578454563701159	4.936102236421723
1.858665671991507	5.338658146964855
1.917459963137651	5.741214057507985
1.9184005802972979	6.1884984025559095
1.9191534062389042	6.546325878594249
1.9494143261746528	6.814696485623003
1.9804437570384015	7.217252396166133
1.9811237668749409	7.5303514376996805
1.9821928255364416	8.022364217252395
2.014040305942652	8.559105431309902
2.01483066318364	8.916932907348244
2.0463994732015016	9.095846645367413
2.047202528948116	9.45367412140575
2.0483072451864515	9.945686900958465
2.0810128425639136	10.39297124600639
2.0816252916659104	10.661341853035143
2.082237921013744	10.929712460063897
2.0828507306604624	11.19808306709265
2.1820823652270755	11.645367412140576
2.183045727403937	12.047923322683706
2.1836882051282536	12.316293929712462
2.1846522762634235	12.718849840255592
2.185831163457327	13.210862619808307
2.220623714253556	13.613418530351439
2.255748799054423	13.926517571884984
2.255970068843575	14.015974440894569
2.435343187173589	13.792332268370608
2.47240914208274	13.568690095846645
2.509916143602259	13.300319488817891
2.547992134751788	13.031948881789138
2.547742222888948	12.942492012779553
2.5472424726963405	12.763578274760384
2.585884712634639	12.49520766773163
2.585123901946731	12.226837060702875
2.5848703481302975	12.13738019169329
2.6649367198252545	11.95846645367412
2.7057624775554308	11.824281150159745
2.7049663968168707	11.555910543130992
2.788889517327686	11.421725239616613
2.8313364519693063	11.19808306709265
2.874711386608766	11.063897763578275
2.9184645312144477	10.840255591054312
2.9630289125374594	10.661341853035143
3.0545091262274338	10.39297124600639
3.054059749243192	10.258785942492011
3.3449347197682875	9.230031948881788
3.1006946429665376	10.079872204472842
3.246029801024168	9.856230031948881
3.2453930788300234	9.677316293929712
3.2951111943071494	9.543130990415335
3.5020613401114877	9.095846645367413
3.447698552067784	8.827476038338656
3.4996576261267096	8.46964856230032
3.6080595382385616	8.29073482428115
3.662614979821021	7.977635782747603
3.775508965117988	7.6645367412140555
3.891882718828082	7.3514376996805115
4.073902675725465	7.038338658146964
4.072304606697792	6.680511182108624
4.26255368274972	6.322683706070286
4.462128429352407	6.054313099041533
4.529597766895492	5.741214057507985
4.6703601576246125	5.651757188498401
4.814552319712819	5.383386581469647
4.887350460940839	5.070287539936102
5.116680337023466	4.891373801916931
5.43963494466012	4.712460063897762
5.522426259101111	4.488817891373801
5.870990736112753	4.309904153354632
6.052547858444415	4.08626198083067
6.434256983067591	3.8626198083067074
6.734851256908489	3.5047923322683694
7.27176846857922	3.460063897763577
7.970598769617529	3.1469648562300296
8.475349463592782	3.1469648562300296
8.60518840547276	3.012779552715653
9.877171346548483	2.7444089456868994
9.433550667216632	2.833865814696484
10.664078238599888	2.654952076677315
11.513112761822988	2.520766773162938
12.055117812022477	2.476038338658146
12.241598338236543	2.476038338658146
13.837733513610797	2.252396166134183
13.420670934728445	2.341853035143769
15.887064673045423	2.2076677316293907
16.633351518195692	2.0734824281150157
19.095769650509038	1.9840255591054294
19.99474491481308	1.9392971246006354
20.93501466425129	1.8498402555910527
20.93501466425129	1.8498402555910527
22.259665654857805	1.8051118210862622
25.163252361956864	1.6261980830670897
25.163252361956864	1.6261980830670897
25.94777399049993	1.6261980830670897
29.33100497427884	1.4025559105431284
29.33100497427884	1.4025559105431284
32.15765303275845	1.3130990415335457
"""
data_array_bump = np.genfromtxt(StringIO(data_str_bump))
R_bump = data_array_bump[:, 0]   # R/Rsun
xi_data = data_array_bump[:, 1]  # T_perp/T_parallel
# バンプモデルの定義（対数座標上のシグモイド）
def bump_model(x, A, D, k1, mu1, k2, mu2):
    logistic_rise = 1.0 / (1 + np.exp(-k1 * (np.log(x) - mu1)))
    logistic_fall = 1.0 / (1 + np.exp(-k2 * (np.log(x) - mu2)))
    return A + D * logistic_rise * (1 - logistic_fall)
initial_guess = [1.3, 13.0, 10.0, 0.3, 10.0, 1.0]
lower_bounds = [0.0,   0.0,  0.0, -5.0,  0.0, -5.0]
upper_bounds = [10.0,  50.0, 50.0,  5.0, 50.0,  5.0]
from scipy.optimize import curve_fit
popt, pcov = curve_fit(bump_model, R_bump, xi_data, p0=initial_guess, bounds=(lower_bounds, upper_bounds), maxfev=10000)
print("バンプモデルの最適化パラメータ:")
print("A =", popt[0])
print("D =", popt[1])
print("k1 =", popt[2])
print("mu1 =", popt[3])
print("k2 =", popt[4])
print("mu2 =", popt[5])
x_fit = np.linspace(np.min(R_bump), np.max(R_bump), 1000)
y_fit = bump_model(x_fit, *popt)
plt.figure(figsize=(8,5))
plt.scatter(R_bump, xi_data, color='red', label='Data')
plt.plot(x_fit, y_fit, label='Fitted Bump Model')
plt.xscale("log")
plt.xlabel("R/Rsun")
plt.ylabel("T_perp/T_parallel")
plt.title("Fitted Bump Model")
plt.legend()
plt.grid(True)
plt.show()

#############################################
# ③ β = 4π p / B^2 を計算してプロット
#############################################
# p = n * (k_B/m_p) * T, ここで T = Tpara * (1 + 2*xi)/3, Tpara は spline_Tpara の結果, xi はバンプモデルから得る
# CGS単位系: k_B = 1.38e-16 erg/K, m_p = 1.67e-24 g
k_b = 1.38e-16       # erg/K
m_p = 1.6726219e-24  # g

# n(R) と B(R) の定義（CGS単位系）
n_0 = 3e8             # r=1 における数密度の初期値（任意の値）
B_0 = 9.7              # r=1 における磁場の初期値（任意の値）
alpha = 2.59         # 数密度の減衰指数
beta_exponent = 1.66 # 磁場の減衰指数

R_target = np.linspace(1, 30, 60)  # R/Rsun
# Tpara_target は spline_Tpara により得た T_parallel の評価値
Tpara_target = spline_Tpara(np.log(R_target))
# xi_target はバンプモデルから評価（T_perp/T_parallel の値）
xi_target = bump_model(R_target, *popt)
# 温度 T の計算: T = Tpara * (1 + 2*xi)/3
T = Tpara_target * (1 + 2 * xi_target) / 3

# n(R) と B(R) の計算
n_R = n_0 * (R_target)**(-alpha)
B_R = B_0 * (R_target)**(-beta_exponent)

# 圧力 p の計算
p = n_R * k_b * T  # erg/cm^3

# β の計算: β = (4π p) / B^2
beta = 8 * np.pi * p / (B_R**2)

plt.figure(figsize=(10,6))
plt.plot(R_target, beta, 'b-', lw=2, label=r'$\beta=\frac{8\pi\,p}{B^2}$')
plt.xlabel(r'$R/R_\odot$', fontsize=18)
plt.ylabel(r'$\beta$', fontsize=18)
plt.title(r'$\beta$ vs $R/R_\odot$', fontsize=20)
plt.legend(fontsize=16)
plt.grid(True, which='both', ls='--')
plt.tight_layout()
plt.show()

#############################################
# ④ PDI 成長率の計算（β の値を利用）
#############################################
# ここでは、先に補間したβの値を使って、各 r_ratio での PDI 成長率（最大虚部）を求める
# （下記の式は例示用です）

# まず、R_target, beta の配列から補間関数を作成
B_perp_squared_0=0.001
beta_interp = UnivariateSpline(R_target, beta, s=0)
# r=1 のときの β の値
beta_at_1 = beta_interp(1)

def calculate_max_imaginary_part(r_ratio):
    # 補間関数からその r_ratio での β を取得
    beta_val = beta_interp(r_ratio)
    # B_perp_squared の r によるスケーリング（例として r^1 を用いる）
    B_perp_squared = B_perp_squared_0 * r_ratio**(0.92)

    # ここでの式は例示です。以下の Newton 法で用いる f, df は適当な例です。
    def newton_method(omega_hat, k_hat):
        f = ((omega_hat - k_hat) * (omega_hat**2 - beta_val * k_hat**2) *
             ((omega_hat + k_hat)**2 - 4) -
             (B_perp_squared * k_hat**2) * (omega_hat**3 + k_hat * omega_hat**2 - 3 * omega_hat + k_hat))
        df = (3 * B_perp_squared * k_hat**2 + beta_val * k_hat**4 + 4 * beta_val * k_hat**2 +
              4 * k_hat * omega_hat**3 + 5 * omega_hat**4 +
              3 * omega_hat**2 * (-B_perp_squared * k_hat**2 - beta_val * k_hat**2 - k_hat**2 - 4) +
              2 * omega_hat * (-B_perp_squared * k_hat**3 - beta_val * k_hat**3 - k_hat**3 + 4 * k_hat))
        if np.abs(df) < 1e-10:
            return omega_hat
        return omega_hat - f / df

    def root_of_func(initial_guesses, k_hat):
        roots = []
        for omega_hat in initial_guesses:
            for _ in range(300):
                omega_hat_new = newton_method(omega_hat, k_hat)
                if np.abs(omega_hat_new - omega_hat) < 1e-10:
                    if all(np.abs(omega_hat_new - root) > 1e-10 for root in roots):
                        roots.append(omega_hat_new)
                    break
                omega_hat = omega_hat_new
        return roots

    k_hat_values = np.linspace(0.85, 2, 1000)
    omega_hats_imag = []
    initial_guesses = [0.1+0.2j, 2+0.5j, -5+2j, -5-8j, 7-3j, 1+1j, 2+0.1j, 2-0.1j,
                       -2+2j, -2-2j, 3+3j, 3-3j, -4+0.5j, 4+0.5j, 0+3j, 0-3j,
                       1+2j, 1-2j, -1+0.5j, -1-0.5j, 0.5+1j, 0.5-1j, -0.5+0.2j,
                       -0.5-0.2j, 3.5+4j, 3.5-4j, -3.5+4j, -3.5-4j,
                       2.15+0.1j, 2.20+0.1j, 2.25+0.1j, 2.1+0.1j, 2.15+0.07j, 2.15+0.04j, 2.15+0.01j, -4-4j,
                       6+2j, 6-2j, -6+2j, -6-2j, 0.1+0.3j, 0.1-0.3j, -0.1+0.3j, -0.1-0.3j]

    for k_hat in k_hat_values:
        roots = root_of_func(initial_guesses, k_hat)
        for root in roots:
            omega_hats_imag.append(root.imag)
    return np.max(omega_hats_imag)

# r_ratios の範囲 (1 から 30) を 30 点生成
r_ratios = np.linspace(1, 30, 90)

with Pool() as pool:
    max_imaginary_parts = pool.map(calculate_max_imaginary_part, r_ratios)

plt.figure(figsize=(14, 8))
plt.plot(r_ratios, max_imaginary_parts, lw=2, color='tab:blue',
         label=f"$\\beta(r=1) = {beta_at_1:.3e}$")
plt.xlabel(r"$r/r_{0}$", fontsize=25)
plt.ylabel(r"$\gamma_{max} / \omega_{0}$", fontsize=25)
plt.title("Maximum Growth Rate (MHD) vs r", fontsize=25)
plt.tick_params(axis='both', labelsize=20)
plt.legend(fontsize=16)
plt.grid(True, which='both', ls='--')
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
from io import StringIO
from multiprocessing import Pool

# ======== 既存のデータ読み込みや補間、フィッティングのコード ===========
# （ここでは既に R_target, spline_Tpara, bump_model, popt, beta_para_interp, B_perp_squared_0, beta_interp などが定義されているものとします）

# ----- 例：β の補間関数（MHD計算用） -----
# ※ この beta_interp は、先のコードで作成した R_target, beta のスプライン補間関数です
# 例: beta_interp = UnivariateSpline(R_target, beta, s=0)
# --------------------------------------------------------------------------

# ======== MHD版 最大成長率計算用関数 ========
def calculate_max_imaginary_part_mhd(r_ratio):
    beta_val = beta_interp(r_ratio)  # 例：beta_interp = UnivariateSpline(R_target, beta, s=0)
    B_perp_squared = B_perp_squared_0 * r_ratio**(0.92)

    def newton_method(omega_hat, k_hat):
        f = ((omega_hat - k_hat) * (omega_hat**2 - beta_val * k_hat**2) *
             ((omega_hat + k_hat)**2 - 4) -
             (B_perp_squared * k_hat**2) * (omega_hat**3 + k_hat * omega_hat**2 - 3 * omega_hat + k_hat))
        df = (3 * B_perp_squared * k_hat**2 + beta_val * k_hat**4 + 4 * beta_val * k_hat**2 +
              4 * k_hat * omega_hat**3 + 5 * omega_hat**4 +
              3 * omega_hat**2 * (-B_perp_squared * k_hat**2 - beta_val * k_hat**2 - k_hat**2 - 4) +
              2 * omega_hat * (-B_perp_squared * k_hat**3 - beta_val * k_hat**3 - k_hat**3 + 4 * k_hat))
        if np.abs(df) < 1e-10:
            return omega_hat
        return omega_hat - f / df

    def root_of_func(initial_guesses, k_hat):
        roots = []
        for omega_hat in initial_guesses:
            for _ in range(300):
                omega_hat_new = newton_method(omega_hat, k_hat)
                if np.abs(omega_hat_new - omega_hat) < 1e-10:
                    if all(np.abs(omega_hat_new - root) > 1e-10 for root in roots):
                        roots.append(omega_hat_new)
                    break
                omega_hat = omega_hat_new
        return roots

    k_hat_values = np.linspace(0.85, 2, 1000)
    omega_hats_imag = []
    initial_guesses = [0.1+0.2j, 2+0.5j, -5+2j, -5-8j, 7-3j, 1+1j, 2+0.1j, 2-0.1j,
                       -2+2j, -2-2j, 3+3j, 3-3j, -4+0.5j, 4+0.5j, 0+3j, 0-3j,
                       1+2j, 1-2j, -1+0.5j, -1-0.5j, 0.5+1j, 0.5-1j, -0.5+0.2j,
                       -0.5-0.2j, 3.5+4j, 3.5-4j, -3.5+4j, -3.5-4j,
                       2.15+0.1j, 2.20+0.1j, 2.25+0.1j, 2.1+0.1j, 2.15+0.07j, 2.15+0.04j, 2.15+0.01j, -4-4j,
                       6+2j, 6-2j, -6+2j, -6-2j, 0.1+0.3j, 0.1-0.3j, -0.1+0.3j, -0.1-0.3j]
    for k_hat in k_hat_values:
        roots = root_of_func(initial_guesses, k_hat)
        for root in roots:
            omega_hats_imag.append(root.imag)
    return np.max(omega_hats_imag)

# ======== PDI版 最大成長率計算用関数 ========
def calculate_max_imaginary_part_pdi(r_ratio):
    beta_parallel = beta_para_interp(r_ratio)
    xi = bump_model(r_ratio, *popt)
    B_perp_squared = B_perp_squared_0 * r_ratio**(0.92)
    tilde_beta = beta_parallel * (3.0/2.0) / (1.0 + B_perp_squared + beta_parallel/2.0 * (xi - 1.0))

    def newton_method(omega_hat, k_hat):
        f = ((omega_hat**2 - tilde_beta * k_hat**2 * (1.0 + B_perp_squared * xi / 3.0)) *
             ((omega_hat - k_hat) * ((omega_hat + k_hat)**2 - 4.0) +
              tilde_beta * B_perp_squared * (xi - 4.0) / (3.0*(1.0+B_perp_squared)) *
             ((k_hat**2+1.0)*omega_hat + k_hat*(k_hat**2-3.0))) -
             (B_perp_squared * k_hat**2 * (1.0 - tilde_beta*(3.0-xi-B_perp_squared) / (3.0*(1.0+B_perp_squared))) *
              ((omega_hat**3+omega_hat**2*k_hat-3.0*omega_hat+k_hat) -
               tilde_beta*(3.0-xi)/3.0 * ((k_hat**2+1.0)*omega_hat+k_hat*(k_hat**2-3.0)))))
        df = (2*omega_hat * (((omega_hat-k_hat)*((omega_hat+k_hat)**2-4.0)) +
              tilde_beta*B_perp_squared*(xi-4.0)/(3.0*(1.0+B_perp_squared)) *
              ((k_hat**2+1.0)*omega_hat+k_hat*(k_hat**2-3.0))) +
              (omega_hat**2 - tilde_beta*k_hat**2*(1.0+B_perp_squared*xi/3.0)) *
              (((omega_hat+k_hat)**2-4.0) + (omega_hat-k_hat)*2*(omega_hat+k_hat)) +
              tilde_beta*B_perp_squared*(xi-4.0)/(3.0*(1.0+B_perp_squared))*(k_hat**2+1.0) -
              B_perp_squared*k_hat**2*(1.0 - tilde_beta*(3.0-xi-B_perp_squared) / (3.0*(1.0+B_perp_squared))) *
              (3*omega_hat**2+2*omega_hat*k_hat-3.0) -
              (B_perp_squared*k_hat**2 * (-tilde_beta*(3.0-xi)/3.0))*(k_hat**2+1.0)
             )
        if np.abs(df) < 1e-10:
            return omega_hat
        return omega_hat - f/df

    def root_of_func(initial_guesses, k_hat):
        roots = []
        for omega_hat in initial_guesses:
            for _ in range(1000):
                omega_hat_new = newton_method(omega_hat, k_hat)
                if np.abs(omega_hat_new - omega_hat) < 1e-10:
                    if all(np.abs(omega_hat_new - root) > 1e-10 for root in roots):
                        roots.append(omega_hat_new)
                    break
                omega_hat = omega_hat_new
        return roots

    k_hat_values = np.linspace(0.85, 2, 1000)
    omega_hats_imag = []
    initial_guesses = [0.1+0.2j, 2+0.5j, -5+2j, -5-8j, 7-3j, 1+1j, 2+0.1j, 2-0.1j,
                       -2+2j, -2-2j, 3+3j, 3-3j, -4+0.5j, 4+0.5j, 0+3j, 0-3j,
                       1+2j, 1-2j, -1+0.5j, -1-2j, 0.5+1j, 0.5-1j, -0.5+0.2j,
                       -0.5-0.2j, 3.5+4j, 3.5-4j, -3.5+4j, -3.5-4j,
                       2.15+0.1j, 2.20+0.1j, 2.25+0.1j, 2.1+0.1j, 2.15+0.07j, 2.15+0.04j, 2.15+0.01j, -4-4j,
                       6+2j, 6-2j, -6+2j, -6-2j, 0.1+0.3j, 0.1-0.3j, -0.1+0.3j, -0.1-0.3j]
    for k_hat in k_hat_values:
        roots = root_of_func(initial_guesses, k_hat)
        for root in roots:
            omega_hats_imag.append(root.imag)
    return np.max(omega_hats_imag)

# ======== 両者の計算と同一Figureでのプロット ============
# 共通の r_ratio の範囲（例：1～30）
r_ratios = np.linspace(1, 30, 90)

# MHD版最大成長率の計算
with Pool() as pool:
    max_imag_mhd = pool.map(calculate_max_imaginary_part_mhd, r_ratios)

# PDI版の場合は Firehose 条件により計算可能な範囲を制限する（例）
def calculate_firehose_condition(r):
    beta_parallel = beta_para_interp(r)
    xi = bump_model(r, *popt)
    B_perp_squared = B_perp_squared_0 * r**(0.92)
    return 1 + 0.5 * beta_parallel * (xi - 1) / (1 + B_perp_squared)

firehose_values = np.array([calculate_firehose_condition(r) for r in r_ratios])
cutoff_index = np.argmax(firehose_values < 0) if np.any(firehose_values < 0) else len(r_ratios)
r_ratios_pdi = r_ratios[:cutoff_index]

with Pool() as pool:
    max_imag_pdi = pool.map(calculate_max_imaginary_part_pdi, r_ratios_pdi)

# プロット
plt.rcParams["font.size"] = 20
plt.figure(figsize=(14, 8))
plt.plot(r_ratios, max_imag_mhd, lw=2, color='tab:blue', label=r"$\gamma_{max}/\omega_{0} (isotropic)$")
plt.plot(r_ratios_pdi, max_imag_pdi, lw=2, color='tab:red', label=r"$\gamma_{max}/\omega_{0}$ (CGL)")
plt.xlabel(r"$R/R_{0}$", fontsize=25)
plt.ylabel(r"$\gamma_{max}/\omega_{0}$", fontsize=25)
plt.title("$\gamma_{max}/\omega_{0}$ vs r", fontsize=25)
plt.legend(fontsize=18)
plt.grid(True)
plt.tight_layout()
plt.show()

# ▼ 追加セル ──────────────────────────────────────────────
# ---------------------------------------------------------
# 1) 物理パラメータを R 方向に並べた配列を準備
# ---------------------------------------------------------
beta_iso_vals       = beta_interp(r_ratios)              # 等方 MHD の β
beta_parallel_vals  = beta_para_interp(r_ratios)         # CGL β||
xi_vals             = bump_model(r_ratios, *popt)        # CGL ξ
B_perp_vals         = B_perp_squared_0 * r_ratios**(0.92)     # B⊥² (r¹ スケーリング)

# ---------------------------------------------------------
# 2) 連結プロット作成
# ---------------------------------------------------------
fig, axes = plt.subplots(
    nrows=2, ncols=1, sharex=True, figsize=(14, 10),
    gridspec_kw={'hspace': 0.04, 'height_ratios': [1.5, 1]}
)

# ── (i) 上段：成長率比較 ──────────────────────────────
ax_top = axes[0]
ax_top.plot(r_ratios,      max_imag_mhd, lw=2, color='tab:blue',
            label=r'$\gamma_{\max}/\omega_{0}\;$(isotropic)')
ax_top.plot(r_ratios_pdi,  max_imag_pdi, lw=2, color='tab:red',
            label=r'$\gamma_{\max}/\omega_{0}\;$(CGL)')

ax_top.set_ylabel(r'$\gamma_{\max}/\omega_{0}$')
ax_top.grid(True, alpha=0.3)
ax_top.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

# ── (ii) 下段：β, β∥, ξ, 𝐵⊥² ─────────────────────────
ax_bot = axes[1]

ax_bot.plot(r_ratios, beta_iso_vals,      lw=2, color='tab:brown',
            label=r'$\beta$ (isotropic)')
ax_bot.plot(r_ratios, beta_parallel_vals, lw=2, color='tab:purple',
            label=r'$\beta_{\parallel}$ (CGL)')
ax_bot.plot(r_ratios, xi_vals,            lw=2, color='tab:orange',
            label=r'$\xi=T_{\perp}/T_{\parallel}$')
ax_bot.plot(r_ratios, B_perp_vals,        lw=2, color='tab:cyan',
            label=r'$\hat{B}_{\perp}^{2}$')

ax_bot.set_yscale('log')
ax_bot.set_xlabel(r'$R/R_{0}$')
ax_bot.set_ylabel(r'$\beta,\;\beta_{\parallel},\;\xi,\;\hat{B}_{\perp}^{2}$')
ax_bot.set_ylim(1e-5, 1e2)
ax_bot.set_yticks([1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2])
from matplotlib.ticker import LogLocator
# ───── log‐grid 追加ここから ─────
ax_bot.yaxis.set_minor_locator(LogLocator(base=10.0,          # 10 のべき
                                       subs=np.arange(1, 10)*0.1,  # 1–9 の副目盛り
                                       numticks=100))
ax_bot.grid(True,  which='major', alpha=0.6)  # メジャー
ax_bot.grid(True,  which='minor', axis='y', alpha=0.3)  # マイナー
# ───── ここまで ─────
ax_bot.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

# ── レイアウト調整 ──────────────────────────────────
plt.subplots_adjust(right=0.8)
plt.tight_layout()
plt.show()
# ▲ 追加セル ──────────────────────────────────────────────

# ▼▼▼ ここから追記 ─────────────────────────────────────────
# -----------------------------------------------------------
# 6. データセット（90 × 7）を作成して CSV に保存
#     0: r_ratio
#     1: γ_max/ω0  (CGL - PDI)           … Firehoseが負になる r では NaN
#     2: γ_max/ω0  (isotropic-MHD)
#     3: β         (isotropic-MHD)
#     4: β_parallel(CGL - PDI)
#     5: ξ         (=T_perp/T_parallel)
#     6: B_perp²   (≡ B_perp_squared_0 · r^0.92)
# -----------------------------------------------------------

# ---- ① CGL 成長率を全 r 90 点に並べる（Firehoseカット以降は NaN） ----
gamma_CGL_full = np.full_like(r_ratios, np.nan, dtype=float)
gamma_CGL_full[:cutoff_index] = max_imag_pdi   # PDI 計算で得た値を先頭から代入

# ---- ② 列方向にスタックしてデータセット完成 ----------------------------
dataset = np.column_stack([
    r_ratios,           # 0
    gamma_CGL_full,     # 1
    max_imag_mhd,       # 2
    beta_iso_vals,      # 3
    beta_parallel_vals, # 4
    xi_vals,            # 5
    B_perp_vals         # 6
])

# ---- ③ CSV 出力 ---------------------------------------------------------
header = ("R_over_R0, gamma_CGL, gamma_isotropic, "
          "beta_isotropic, beta_parallel, xi, B_perp_squared")

np.savetxt(
    "Fig4b_dataset_full.csv",
    dataset,
    delimiter=",",
    header=header,
    comments=""
)

print(">> Fig4b_dataset_full.csv を保存しました")
print("\n--- preview ---")
print(dataset[:5])   # 先頭5行を確認
# ▲▲▲ ここまで追記 ─────────────────────────────────────────


#シナリオ４＆５　beta=0.1

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
from io import StringIO
from multiprocessing import Pool
import sympy as sp

#############################
# ① βₚₐᵣₐ の計算（Tₚₐᵣₐ データから）
#############################

# --- Tₚₐᵣₐ のデータ定義 ---
data_str_Tpara = """
0.9363634313172465	630950.7344480193
0.9518215986713207	735640.22544596407
0.952918054833259	874310.24580220731
0.9679063371940468	908510.75756516862
0.998210703186863	926110.87281287937
1.0136520170010803	926110.87281287937
1.110736672952074	841390.51416451947
1.2356312656535287	735640.22544596407
1.3529376348534548	595660.214352901094
1.5242477252204125	348070.00588428406
1.5691579687708173	271220.725793320296
1.6153914420970539	211340.89039836648
1.6640511310707182	181270.308811846968
1.7680722177343189	161550.98098439873
1.820865124259488	133350.21432163324
2.0274233521107727	133350.21432163324
2.294386965053555	152520.229565390198
2.4434305526939717	192010.41938638801
2.6014903677573367	232630.05067153624
2.856128729537577	281830.82931264455
3.0404960563039145	334960.54391578279
3.3897421734448465	405810.98942243799
3.665782272509721	510890.69774506924
3.9627803425543924	607200.21956909884
4.6977390496362625	735640.22544596407
5.484172555736265	891250.0938133744
6.1125471512033736	1039120.2303835169
7.241580008400507	1143750.58630495366
8.447383717827934	1234990.91726875826
10.966184681342897	1234990.91726875826
12.58764378600493	1188500.22274370166
15.127869790847797	1122010.84543019629
18.180721372437116	1059250.37251772876
21.18632136291081	980990.47127268763
25.471560675002788	980990.47127268763
30.15705228022852	980990.47127268763
1.3518996152492362	530880.44442309879
1.4147075349881577	482310.78482239309
1.4580679970023382	446680.35921509626
1.4796761259208662	405810.98942243799
1.523078270709203	310210.776647994982
1.5679540575987572	241730.154808041014
1.2539429491545022	668340.39175686135
1.766941707999398	146770.992676220705
2.1566406048857756	141250.375446227554
2.3680392085234327	174440.826989992558
2.4828154797631	211340.89039836648
2.6853444456585054	271220.725793320296
2.993030283874072	316220.776601683792
3.235108784158188	368690.450645195735
3.4443804148047565	446680.35921509626
4.152726538879247	681290.20690579608
4.922913696391965	825400.41852680173
9.405651624378091	1234990.91726875826
6.603554671772791	1122010.84543019629
5.745572820257177	962350.06263980868
13.59182511235687	1188500.22274370166
16.33260863538179	1100690.4171252208
22.87939016505923	1000000
28.364672776598606	1000000
19.0399878392478	1079770.51623277094
19.3295713029763	1039120.2303835169
7.821278990952227	1188500.22274370166
1.1449269489307676	794320.82347242805
32.070897303318624	1000000
34.62492878921517	980990.47127268763
37.965479460384884	980990.47127268763
42.26686142656028	962350.06263980868
47.04354057259765	908510.75756516862
"""

# --- 文字列からデータを読み込み ---
data_Tpara = np.genfromtxt(data_str_Tpara.strip().splitlines())
R_Tpara = data_Tpara[:, 0]         # R/Rsun
Tpara = data_Tpara[:, 1]           # T_para

# --- R の昇順にソート ---
idx = np.argsort(R_Tpara)
R_Tpara = R_Tpara[idx]
Tpara = Tpara[idx]

# --- R 軸の対数変換 ---
log_R_Tpara = np.log(R_Tpara)

# --- スプライン補間（smoothing_parameter は適宜調整） ---
smoothing_parameter = 1e11
spline_Tpara = UnivariateSpline(log_R_Tpara, Tpara, s=smoothing_parameter)

# --- R/Rsun = 1 から 30 の範囲を 60 分割 ---
R_target = np.linspace(1, 30, 60)

# --- 対数変換してスプライン関数で T_para を評価 ---
T_target = spline_Tpara(np.log(R_target))

# --- 定数および初期値 ---
k_b = 1.38e-16        # ボルツマン定数 [J/K]
n_0 = 1e8             # r=1 における数密度の初期値（任意の値）
B_0 = 1.78              # r=1 における磁場の初期値（任意の値）

# --- Rsun=1 における T_para ---
Tpara_0 = spline_Tpara(np.log(1))
print("Rsun=1 のときの T_para =", Tpara_0)

# --- R に依存する n(R) と B(R) の定義 ---
n_R = n_0 * (R_target)**(-2.59)
B_R = B_0 * (R_target)**(-1.66)

# --- β_parallel の計算 ---
beta_para = 8 * np.pi * n_R * k_b * T_target / (B_R**2)

# --- β_parallel のプロット（参考） ---
plt.figure(figsize=(10,6))
plt.plot(R_target, beta_para, 'b-', lw=2, label=r'$\beta_{\parallel}$')
plt.xlabel(r'$R/R_\odot$')
plt.ylabel(r'$\beta_{\parallel}$')
plt.title(r'$\beta_{\parallel}$ vs $R/R_\odot$')
plt.legend()
plt.grid(True)
plt.show()

# --- 補間関数を作成（PDI 用に β_parallel を任意の r で評価） ---
beta_para_interp = UnivariateSpline(R_target, beta_para, s=0)

#############################
# ② T_perp/T_para（=xi）のバンプモデルフィッティング
#############################

# --- T_perp/T_para のデータ定義 ---
data_str_bump = """
1.0009813481265506	0.910543130990412
1.0810468494787815	1.0894568690095827
1.149562217230484	1.1341853035143732
1.2413904876306563	1.2236421725239612
1.2998954188768872	1.2236421725239612
1.3611576025865708	1.2236421725239612
1.4036636255995907	1.2683706070287517
1.4698162542259607	1.2683706070287517
1.5396905343022556	1.6261980830670897
1.6373544520272827	1.7156549520766742
1.6630904432580889	1.9392971246006354
1.715782341439801	2.3865814696485597
1.7428365372600465	2.654952076677315
1.7436059778765893	3.057507987220447
1.7988488596018384	3.5047923322683694
1.7992900215245589	3.7284345047923306
1.8003492516192892	4.265175718849839
1.8288265752326527	4.578274760383385
1.8578454563701159	4.936102236421723
1.858665671991507	5.338658146964855
1.917459963137651	5.741214057507985
1.9184005802972979	6.1884984025559095
1.9191534062389042	6.546325878594249
1.9494143261746528	6.814696485623003
1.9804437570384015	7.217252396166133
1.9811237668749409	7.5303514376996805
1.9821928255364416	8.022364217252395
2.014040305942652	8.559105431309902
2.01483066318364	8.916932907348244
2.0463994732015016	9.095846645367413
2.047202528948116	9.45367412140575
2.0483072451864515	9.945686900958465
2.0810128425639136	10.39297124600639
2.0816252916659104	10.661341853035143
2.082237921013744	10.929712460063897
2.0828507306604624	11.19808306709265
2.1820823652270755	11.645367412140576
2.183045727403937	12.047923322683706
2.1836882051282536	12.316293929712462
2.1846522762634235	12.718849840255592
2.185831163457327	13.210862619808307
2.220623714253556	13.613418530351439
2.255748799054423	13.926517571884984
2.255970068843575	14.015974440894569
2.435343187173589	13.792332268370608
2.47240914208274	13.568690095846645
2.509916143602259	13.300319488817891
2.547992134751788	13.031948881789138
2.547742222888948	12.942492012779553
2.5472424726963405	12.763578274760384
2.585884712634639	12.49520766773163
2.585123901946731	12.226837060702875
2.5848703481302975	12.13738019169329
2.6649367198252545	11.95846645367412
2.7057624775554308	11.824281150159745
2.7049663968168707	11.555910543130992
2.788889517327686	11.421725239616613
2.8313364519693063	11.19808306709265
2.874711386608766	11.063897763578275
2.9184645312144477	10.840255591054312
2.9630289125374594	10.661341853035143
3.0545091262274338	10.39297124600639
3.054059749243192	10.258785942492011
3.3449347197682875	9.230031948881788
3.1006946429665376	10.079872204472842
3.246029801024168	9.856230031948881
3.2453930788300234	9.677316293929712
3.2951111943071494	9.543130990415335
3.5020613401114877	9.095846645367413
3.447698552067784	8.827476038338656
3.4996576261267096	8.46964856230032
3.6080595382385616	8.29073482428115
3.662614979821021	7.977635782747603
3.775508965117988	7.6645367412140555
3.891882718828082	7.3514376996805115
4.073902675725465	7.038338658146964
4.072304606697792	6.680511182108624
4.26255368274972	6.322683706070286
4.462128429352407	6.054313099041533
4.529597766895492	5.741214057507985
4.6703601576246125	5.651757188498401
4.814552319712819	5.383386581469647
4.887350460940839	5.070287539936102
5.116680337023466	4.891373801916931
5.43963494466012	4.712460063897762
5.522426259101111	4.488817891373801
5.870990736112753	4.309904153354632
6.052547858444415	4.08626198083067
6.434256983067591	3.8626198083067074
6.734851256908489	3.5047923322683694
7.27176846857922	3.460063897763577
7.970598769617529	3.1469648562300296
8.475349463592782	3.1469648562300296
8.60518840547276	3.012779552715653
9.877171346548483	2.7444089456868994
9.433550667216632	2.833865814696484
10.664078238599888	2.654952076677315
11.513112761822988	2.520766773162938
12.055117812022477	2.476038338658146
12.241598338236543	2.476038338658146
13.837733513610797	2.252396166134183
13.420670934728445	2.341853035143769
15.887064673045423	2.2076677316293907
16.633351518195692	2.0734824281150157
19.095769650509038	1.9840255591054294
19.99474491481308	1.9392971246006354
20.93501466425129	1.8498402555910527
20.93501466425129	1.8498402555910527
22.259665654857805	1.8051118210862622
25.163252361956864	1.6261980830670897
25.163252361956864	1.6261980830670897
25.94777399049993	1.6261980830670897
29.33100497427884	1.4025559105431284
29.33100497427884	1.4025559105431284
32.15765303275845	1.3130990415335457
"""

# 文字列から数値データを読み込む
data_array_bump = np.genfromtxt(StringIO(data_str_bump))
R_bump = data_array_bump[:, 0]  # R/Rsun
xi_data = data_array_bump[:, 1] # T_perp/T_para

# バンプモデルの定義（対数座標上のシグモイド）
def bump_model(x, A, D, k1, mu1, k2, mu2):
    logistic_rise = 1.0 / (1 + np.exp(-k1 * (np.log(x) - mu1)))
    logistic_fall = 1.0 / (1 + np.exp(-k2 * (np.log(x) - mu2)))
    return A + D * logistic_rise * (1 - logistic_fall)

# 初期推定値とパラメータ境界
initial_guess = [1.3, 13.0, 10.0, 0.3, 10.0, 1.0]
lower_bounds = [0.0,   0.0,  0.0, -5.0,  0.0, -5.0]
upper_bounds = [10.0,  50.0, 50.0,  5.0, 50.0,  5.0]

# フィッティング実行
popt, pcov = curve_fit(
    bump_model, R_bump, xi_data,
    p0=initial_guess,
    bounds=(lower_bounds, upper_bounds),
    maxfev=10000
)

print("バンプモデルの最適化パラメータ:")
print("A =", popt[0])
print("D =", popt[1])
print("k1 =", popt[2])
print("mu1 =", popt[3])
print("k2 =", popt[4])
print("mu2 =", popt[5])

# フィッティング結果のプロット（参考）
x_fit = np.linspace(np.min(R_bump), np.max(R_bump), 1000)
y_fit = bump_model(x_fit, *popt)
plt.figure(figsize=(8,5))
plt.scatter(R_bump, xi_data, color='red', label='Data')
plt.plot(x_fit, y_fit, label='Fitted Bump Model')
plt.xscale("log")
plt.xlabel("R/Rsun")
plt.ylabel("T_perp/T_para")
plt.title("Fitted Bump Model")
plt.legend()
plt.grid(True)
plt.show()

#############################
# ③ PDI 成長率と Firehose 条件の計算・プロット
#############################

# 固定定数（r=1 における B_perp² の値）
B_perp_squared_0 = 0.001

# ※ ここでは T_perp/T_para = xi をバンプモデルから得るので，
#    固定値 xi0 は不要となります。

def calculate_max_imaginary_part(r_ratio):
    # beta_parallel は補間関数から取得
    beta_parallel = beta_para_interp(r_ratio)
    # xi を固定のスケーリングではなく，バンプモデルから求める．
    xi = bump_model(r_ratio, *popt)
    # ※ B_perp_squared の r によるスケーリングはここでは例として r^1 を用いる（必要に応じて調整）
    B_perp_squared = B_perp_squared_0 * r_ratio**(0.92)
    tilde_beta = beta_parallel * (3.0/2.0) / (1.0 + B_perp_squared + beta_parallel/2.0 * (xi - 1.0))

    def newton_method(omega_hat, k_hat):
        f = ((omega_hat**2 - tilde_beta * k_hat**2 * (1.0 + B_perp_squared * xi / 3.0)) *
             ((omega_hat - k_hat) * ((omega_hat + k_hat)**2 - 4.0) +
              tilde_beta * B_perp_squared * (xi - 4.0) / (3.0*(1.0+B_perp_squared)) *
             ((k_hat**2+1.0)*omega_hat + k_hat*(k_hat**2-3.0))) -
             (B_perp_squared * k_hat**2 * (1.0 - tilde_beta*(3.0-xi-B_perp_squared) / (3.0*(1.0+B_perp_squared))) *
              ((omega_hat**3+omega_hat**2*k_hat-3.0*omega_hat+k_hat) -
               tilde_beta*(3.0-xi)/3.0 * ((k_hat**2+1.0)*omega_hat+k_hat*(k_hat**2-3.0)))))
        df = (2*omega_hat * (((omega_hat-k_hat)*((omega_hat+k_hat)**2-4.0)) +
              tilde_beta*B_perp_squared*(xi-4.0)/(3.0*(1.0+B_perp_squared)) *
              ((k_hat**2+1.0)*omega_hat+k_hat*(k_hat**2-3.0))) +
              (omega_hat**2 - tilde_beta*k_hat**2*(1.0+B_perp_squared*xi/3.0)) *
              (((omega_hat+k_hat)**2-4.0) + (omega_hat-k_hat)*2*(omega_hat+k_hat)) +
              tilde_beta*B_perp_squared*(xi-4.0)/(3.0*(1.0+B_perp_squared))*(k_hat**2+1.0) -
              B_perp_squared*k_hat**2*(1.0 - tilde_beta*(3.0-xi-B_perp_squared) / (3.0*(1.0+B_perp_squared))) *
              (3*omega_hat**2+2*omega_hat*k_hat-3.0) -
              (B_perp_squared*k_hat**2 * (-tilde_beta*(3.0-xi)/3.0))*(k_hat**2+1.0)
             )
        if np.abs(df) < 1e-10:
            return omega_hat
        return omega_hat - f/df

    def root_of_func(initial_guesses, k_hat):
        roots = []
        for omega_hat in initial_guesses:
            for _ in range(1000):
                omega_hat_new = newton_method(omega_hat, k_hat)
                if np.abs(omega_hat_new - omega_hat) < 1e-10:
                    if all(np.abs(omega_hat_new - root) > 1e-10 for root in roots):
                        roots.append(omega_hat_new)
                    break
                omega_hat = omega_hat_new
        return roots

    # k_hat の範囲を決め，各 k_hat での根の虚部の最大値を求める
    k_hat_values = np.linspace(0.95, 2, 100)
    omega_hats_imag = []
    initial_guesses = [0.1+0.2j, 2+0.5j, -5+2j, -5-8j, 7-3j, 1+1j, 2+0.1j, 2-0.1j,
                       -2+2j, -2-2j, 3+3j, 3-3j, -4+0.5j, 4+0.5j, 0+3j, 0-3j,
                       1+2j, 1-2j, -1+0.5j, -1-0.5j, 0.5+1j, 0.5-1j, -0.5+0.2j,
                       -0.5-0.2j, 3.5+4j, 3.5-4j, -3.5+4j, -3.5-4j,
                       2.15+0.1j, 2.20+0.1j, 2.25+0.1j, 2.1+0.1j, 2.15+0.07j, 2.15+0.04j, 2.15+0.01j,
                       -4-4j, 6+2j, 6-2j, -6+2j, -6-2j, 0.1+0.3j, 0.1-0.3j, -0.1+0.3j, -0.1-0.3j]
    for k_hat in k_hat_values:
        roots = root_of_func(initial_guesses, k_hat)
        for root in roots:
            omega_hats_imag.append(root.imag)
    return np.max(omega_hats_imag)

def calculate_firehose_condition(r_ratios):
    beta_parallel_values = beta_para_interp(r_ratios)
    # xi を各 r で，バンプモデルから求める
    xi_values = bump_model(r_ratios, *popt)
    B_perp_squared_values = B_perp_squared_0 * r_ratios**(0.92)
    condition = 1 + 0.5 * beta_parallel_values * (xi_values - 1) / (1 + B_perp_squared_values)
    return condition

# r_ratio の範囲（1～30）
r_ratios = np.linspace(1, 30, 90)

# Firehose 条件の計算
firehose_condition = calculate_firehose_condition(r_ratios)
cutoff_index = np.argmax(firehose_condition < 0) if np.any(firehose_condition < 0) else len(r_ratios)
r_ratios_cutoff = r_ratios[:cutoff_index]

# 並列計算で各 r_ratio に対する最大虚部を求める
with Pool() as pool:
    max_imaginary_parts = pool.map(calculate_max_imaginary_part, r_ratios_cutoff)

# プロット①：PDI 成長率（最大虚部） vs r_ratio
plt.figure(figsize=(14,8))
plt.plot(r_ratios_cutoff, max_imaginary_parts, lw=2, color='tab:blue',
         label=f"$\\beta_{{||}}(r=1) = {beta_para_interp(1):.3e}$, $\\xi(r=1) = {bump_model(1, *popt):.3f}$")
plt.xlabel(r"$r/r_{0}$", fontsize=20)
plt.ylabel(r"$\gamma/\omega_{0}$", fontsize=20)
plt.title("Maximum Growth Rate (PDI) vs r", fontsize=22)
plt.legend(fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.show()

# プロット②：Firehose 条件 vs r_ratio
plt.figure(figsize=(14,8))
plt.plot(r_ratios, firehose_condition, lw=2, color='tab:green',
         label=f"$\\xi(r=1) = {bump_model(1, *popt):.3f}$, $\\beta_{{||}}(r=1) = {beta_para_interp(1):.3e}$")
plt.axhline(0, color='black', lw=1, ls='--')
plt.xlabel(r"$r/r_{0}$", fontsize=20)
plt.ylabel("Discriminant Value", fontsize=20)
plt.title("Firehose Condition vs r", fontsize=22)
plt.xlim(1, 30)
plt.ylim(-1, 1.5)
plt.legend(fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
from io import StringIO
from multiprocessing import Pool
import sympy as sp

# フォントサイズの設定
plt.rcParams.update({'font.size': 16})

#############################################
# ① T_parallel のスプライン補間によるフィッティング
#############################################
data_str_Tpara = """
0.9363634313172465	630950.7344480193
0.9518215986713207	735640.22544596407
0.952918054833259	874310.24580220731
0.9679063371940468	908510.75756516862
0.998210703186863	926110.87281287937
1.0136520170010803	926110.87281287937
1.110736672952074	841390.51416451947
1.2356312656535287	735640.22544596407
1.3529376348534548	595660.214352901094
1.5242477252204125	348070.00588428406
1.5691579687708173	271220.725793320296
1.6153914420970539	211340.89039836648
1.6640511310707182	181270.308811846968
1.7680722177343189	161550.98098439873
1.820865124259488	133350.21432163324
2.0274233521107727	133350.21432163324
2.294386965053555	152520.229565390198
2.4434305526939717	192010.41938638801
2.6014903677573367	232630.05067153624
2.856128729537577	281830.82931264455
3.0404960563039145	334960.54391578279
3.3897421734448465	405810.98942243799
3.665782272509721	510890.69774506924
3.9627803425543924	607200.21956909884
4.6977390496362625	735640.22544596407
5.484172555736265	891250.0938133744
6.1125471512033736	1039120.2303835169
7.241580008400507	1143750.58630495366
8.447383717827934	1234990.91726875826
10.966184681342897	1234990.91726875826
12.58764378600493	1188500.22274370166
15.127869790847797	1122010.84543019629
18.180721372437116	1059250.37251772876
21.18632136291081	980990.47127268763
25.471560675002788	980990.47127268763
30.15705228022852	980990.47127268763
1.3518996152492362	530880.44442309879
1.4147075349881577	482310.78482239309
1.4580679970023382	446680.35921509626
1.4796761259208662	405810.98942243799
1.523078270709203	310210.776647994982
1.5679540575987572	241730.154808041014
1.2539429491545022	668340.39175686135
1.766941707999398	146770.992676220705
2.1566406048857756	141250.375446227554
2.3680392085234327	174440.826989992558
2.4828154797631	211340.89039836648
2.6853444456585054	271220.725793320296
2.993030283874072	316220.776601683792
3.235108784158188	368690.450645195735
3.4443804148047565	446680.35921509626
4.152726538879247	681290.20690579608
4.922913696391965	825400.41852680173
9.405651624378091	1234990.91726875826
6.603554671772791	1122010.84543019629
5.745572820257177	962350.06263980868
13.59182511235687	1188500.22274370166
16.33260863538179	1100690.4171252208
22.87939016505923	1000000
28.364672776598606	1000000
19.0399878392478	1079770.51623277094
19.3295713029763	1039120.2303835169
7.821278990952227	1188500.22274370166
1.1449269489307676	794320.82347242805
32.070897303318624	1000000
34.62492878921517	980990.47127268763
37.965479460384884	980990.47127268763
42.26686142656028	962350.06263980868
47.04354057259765	908510.75756516862
"""
# データ読み込み
data_Tpara = np.genfromtxt(StringIO(data_str_Tpara))
R_Tpara = data_Tpara[:, 0]         # R/Rsun
Tpara = data_Tpara[:, 1]           # T_parallel [K]
idx = np.argsort(R_Tpara)
R_Tpara = R_Tpara[idx]
Tpara = Tpara[idx]
log_R_Tpara = np.log(R_Tpara)
smoothing_parameter = 1e11
spline_Tpara = UnivariateSpline(log_R_Tpara, Tpara, s=smoothing_parameter)
R_target = np.linspace(1, 30, 60)
Tpara_target = spline_Tpara(np.log(R_target))
print("R/Rsun (60点):")
print(R_target)
print("\n対応する T_parallel:")
print(Tpara_target)
plt.figure(figsize=(10, 6))
plt.plot(R_Tpara, Tpara, 'ro', label='data')
plt.plot(R_target, Tpara_target, 'gs', label='evaluated points')
plt.plot(np.exp(np.linspace(log_R_Tpara.min(), log_R_Tpara.max(), 2000)),
         spline_Tpara(np.linspace(log_R_Tpara.min(), log_R_Tpara.max(), 2000)),
         'b-', lw=2, label='smooth fitting spline')
plt.xlabel('R/Rsun (log scale)')
plt.ylabel('T_parallel')
plt.ylim(0,2000000)
plt.title('Smooth fitting spline for R/Rsun and T_parallel\n(評価点: R/Rsun=[1, 30] 60点)')
plt.legend()
plt.grid(True, which='both', ls='--')
plt.show()

#############################################
# ② T_perp/T_parallel のバンプモデルによるフィッティング
#############################################
data_str_bump = """
1.0009813481265506	0.910543130990412
1.0810468494787815	1.0894568690095827
1.149562217230484	1.1341853035143732
1.2413904876306563	1.2236421725239612
1.2998954188768872	1.2236421725239612
1.3611576025865708	1.2236421725239612
1.4036636255995907	1.2683706070287517
1.4698162542259607	1.2683706070287517
1.5396905343022556	1.6261980830670897
1.6373544520272827	1.7156549520766742
1.6630904432580889	1.9392971246006354
1.715782341439801	2.3865814696485597
1.7428365372600465	2.654952076677315
1.7436059778765893	3.057507987220447
1.7988488596018384	3.5047923322683694
1.7992900215245589	3.7284345047923306
1.8003492516192892	4.265175718849839
1.8288265752326527	4.578274760383385
1.8578454563701159	4.936102236421723
1.858665671991507	5.338658146964855
1.917459963137651	5.741214057507985
1.9184005802972979	6.1884984025559095
1.9191534062389042	6.546325878594249
1.9494143261746528	6.814696485623003
1.9804437570384015	7.217252396166133
1.9811237668749409	7.5303514376996805
1.9821928255364416	8.022364217252395
2.014040305942652	8.559105431309902
2.01483066318364	8.916932907348244
2.0463994732015016	9.095846645367413
2.047202528948116	9.45367412140575
2.0483072451864515	9.945686900958465
2.0810128425639136	10.39297124600639
2.0816252916659104	10.661341853035143
2.082237921013744	10.929712460063897
2.0828507306604624	11.19808306709265
2.1820823652270755	11.645367412140576
2.183045727403937	12.047923322683706
2.1836882051282536	12.316293929712462
2.1846522762634235	12.718849840255592
2.185831163457327	13.210862619808307
2.220623714253556	13.613418530351439
2.255748799054423	13.926517571884984
2.255970068843575	14.015974440894569
2.435343187173589	13.792332268370608
2.47240914208274	13.568690095846645
2.509916143602259	13.300319488817891
2.547992134751788	13.031948881789138
2.547742222888948	12.942492012779553
2.5472424726963405	12.763578274760384
2.585884712634639	12.49520766773163
2.585123901946731	12.226837060702875
2.5848703481302975	12.13738019169329
2.6649367198252545	11.95846645367412
2.7057624775554308	11.824281150159745
2.7049663968168707	11.555910543130992
2.788889517327686	11.421725239616613
2.8313364519693063	11.19808306709265
2.874711386608766	11.063897763578275
2.9184645312144477	10.840255591054312
2.9630289125374594	10.661341853035143
3.0545091262274338	10.39297124600639
3.054059749243192	10.258785942492011
3.3449347197682875	9.230031948881788
3.1006946429665376	10.079872204472842
3.246029801024168	9.856230031948881
3.2453930788300234	9.677316293929712
3.2951111943071494	9.543130990415335
3.5020613401114877	9.095846645367413
3.447698552067784	8.827476038338656
3.4996576261267096	8.46964856230032
3.6080595382385616	8.29073482428115
3.662614979821021	7.977635782747603
3.775508965117988	7.6645367412140555
3.891882718828082	7.3514376996805115
4.073902675725465	7.038338658146964
4.072304606697792	6.680511182108624
4.26255368274972	6.322683706070286
4.462128429352407	6.054313099041533
4.529597766895492	5.741214057507985
4.6703601576246125	5.651757188498401
4.814552319712819	5.383386581469647
4.887350460940839	5.070287539936102
5.116680337023466	4.891373801916931
5.43963494466012	4.712460063897762
5.522426259101111	4.488817891373801
5.870990736112753	4.309904153354632
6.052547858444415	4.08626198083067
6.434256983067591	3.8626198083067074
6.734851256908489	3.5047923322683694
7.27176846857922	3.460063897763577
7.970598769617529	3.1469648562300296
8.475349463592782	3.1469648562300296
8.60518840547276	3.012779552715653
9.877171346548483	2.7444089456868994
9.433550667216632	2.833865814696484
10.664078238599888	2.654952076677315
11.513112761822988	2.520766773162938
12.055117812022477	2.476038338658146
12.241598338236543	2.476038338658146
13.837733513610797	2.252396166134183
13.420670934728445	2.341853035143769
15.887064673045423	2.2076677316293907
16.633351518195692	2.0734824281150157
19.095769650509038	1.9840255591054294
19.99474491481308	1.9392971246006354
20.93501466425129	1.8498402555910527
20.93501466425129	1.8498402555910527
22.259665654857805	1.8051118210862622
25.163252361956864	1.6261980830670897
25.163252361956864	1.6261980830670897
25.94777399049993	1.6261980830670897
29.33100497427884	1.4025559105431284
29.33100497427884	1.4025559105431284
32.15765303275845	1.3130990415335457
"""
data_array_bump = np.genfromtxt(StringIO(data_str_bump))
R_bump = data_array_bump[:, 0]   # R/Rsun
xi_data = data_array_bump[:, 1]  # T_perp/T_parallel
# バンプモデルの定義（対数座標上のシグモイド）
def bump_model(x, A, D, k1, mu1, k2, mu2):
    logistic_rise = 1.0 / (1 + np.exp(-k1 * (np.log(x) - mu1)))
    logistic_fall = 1.0 / (1 + np.exp(-k2 * (np.log(x) - mu2)))
    return A + D * logistic_rise * (1 - logistic_fall)
initial_guess = [1.3, 13.0, 10.0, 0.3, 10.0, 1.0]
lower_bounds = [0.0,   0.0,  0.0, -5.0,  0.0, -5.0]
upper_bounds = [10.0,  50.0, 50.0,  5.0, 50.0,  5.0]
from scipy.optimize import curve_fit
popt, pcov = curve_fit(bump_model, R_bump, xi_data, p0=initial_guess, bounds=(lower_bounds, upper_bounds), maxfev=10000)
print("バンプモデルの最適化パラメータ:")
print("A =", popt[0])
print("D =", popt[1])
print("k1 =", popt[2])
print("mu1 =", popt[3])
print("k2 =", popt[4])
print("mu2 =", popt[5])
x_fit = np.linspace(np.min(R_bump), np.max(R_bump), 1000)
y_fit = bump_model(x_fit, *popt)
plt.figure(figsize=(8,5))
plt.scatter(R_bump, xi_data, color='red', label='Data')
plt.plot(x_fit, y_fit, label='Fitted Bump Model')
plt.xscale("log")
plt.xlabel("R/Rsun")
plt.ylabel("T_perp/T_parallel")
plt.title("Fitted Bump Model")
plt.legend()
plt.grid(True)
plt.show()

#############################################
# ③ β = 4π p / B^2 を計算してプロット
#############################################
# p = n * (k_B/m_p) * T, ここで T = Tpara * (1 + 2*xi)/3, Tpara は spline_Tpara の結果, xi はバンプモデルから得る
# CGS単位系: k_B = 1.38e-16 erg/K, m_p = 1.67e-24 g
k_b = 1.38e-16       # erg/K
m_p = 1.6726219e-24  # g

# n(R) と B(R) の定義（CGS単位系）
n_0 = 1e8             # r=1 における数密度の初期値（任意の値）
B_0 = 1.78              # r=1 における磁場の初期値（任意の値）
alpha = 2.59         # 数密度の減衰指数
beta_exponent = 1.66 # 磁場の減衰指数

R_target = np.linspace(1, 30, 60)  # R/Rsun
# Tpara_target は spline_Tpara により得た T_parallel の評価値
Tpara_target = spline_Tpara(np.log(R_target))
# xi_target はバンプモデルから評価（T_perp/T_parallel の値）
xi_target = bump_model(R_target, *popt)
# 温度 T の計算: T = Tpara * (1 + 2*xi)/3
T = Tpara_target * (1 + 2 * xi_target) / 3

# n(R) と B(R) の計算
n_R = n_0 * (R_target)**(-alpha)
B_R = B_0 * (R_target)**(-beta_exponent)

# 圧力 p の計算
p = n_R * k_b * T  # erg/cm^3

# β の計算: β = (4π p) / B^2
beta = 8 * np.pi * p / (B_R**2)

plt.figure(figsize=(10,6))
plt.plot(R_target, beta, 'b-', lw=2, label=r'$\beta=\frac{8\pi\,p}{B^2}$')
plt.xlabel(r'$R/R_\odot$', fontsize=18)
plt.ylabel(r'$\beta$', fontsize=18)
plt.title(r'$\beta$ vs $R/R_\odot$', fontsize=20)
plt.legend(fontsize=16)
plt.grid(True, which='both', ls='--')
plt.tight_layout()
plt.show()

#############################################
# ④ PDI 成長率の計算（β の値を利用）
#############################################
# ここでは、先に補間したβの値を使って、各 r_ratio での PDI 成長率（最大虚部）を求める
# （下記の式は例示用です）

# まず、R_target, beta の配列から補間関数を作成
B_perp_squared_0=0.001
beta_interp = UnivariateSpline(R_target, beta, s=0)
# r=1 のときの β の値
beta_at_1 = beta_interp(1)

def calculate_max_imaginary_part(r_ratio):
    # 補間関数からその r_ratio での β を取得
    beta_val = beta_interp(r_ratio)
    # B_perp_squared の r によるスケーリング（例として r^1 を用いる）
    B_perp_squared = B_perp_squared_0 * r_ratio**(0.92)

    # ここでの式は例示です。以下の Newton 法で用いる f, df は適当な例です。
    def newton_method(omega_hat, k_hat):
        f = ((omega_hat - k_hat) * (omega_hat**2 - beta_val * k_hat**2) *
             ((omega_hat + k_hat)**2 - 4) -
             (B_perp_squared * k_hat**2) * (omega_hat**3 + k_hat * omega_hat**2 - 3 * omega_hat + k_hat))
        df = (3 * B_perp_squared * k_hat**2 + beta_val * k_hat**4 + 4 * beta_val * k_hat**2 +
              4 * k_hat * omega_hat**3 + 5 * omega_hat**4 +
              3 * omega_hat**2 * (-B_perp_squared * k_hat**2 - beta_val * k_hat**2 - k_hat**2 - 4) +
              2 * omega_hat * (-B_perp_squared * k_hat**3 - beta_val * k_hat**3 - k_hat**3 + 4 * k_hat))
        if np.abs(df) < 1e-10:
            return omega_hat
        return omega_hat - f / df

    def root_of_func(initial_guesses, k_hat):
        roots = []
        for omega_hat in initial_guesses:
            for _ in range(300):
                omega_hat_new = newton_method(omega_hat, k_hat)
                if np.abs(omega_hat_new - omega_hat) < 1e-10:
                    if all(np.abs(omega_hat_new - root) > 1e-10 for root in roots):
                        roots.append(omega_hat_new)
                    break
                omega_hat = omega_hat_new
        return roots

    k_hat_values = np.linspace(0.85, 2, 1000)
    omega_hats_imag = []
    initial_guesses = [0.1+0.2j, 2+0.5j, -5+2j, -5-8j, 7-3j, 1+1j, 2+0.1j, 2-0.1j,
                       -2+2j, -2-2j, 3+3j, 3-3j, -4+0.5j, 4+0.5j, 0+3j, 0-3j,
                       1+2j, 1-2j, -1+0.5j, -1-0.5j, 0.5+1j, 0.5-1j, -0.5+0.2j,
                       -0.5-0.2j, 3.5+4j, 3.5-4j, -3.5+4j, -3.5-4j,
                       2.15+0.1j, 2.20+0.1j, 2.25+0.1j, 2.1+0.1j, 2.15+0.07j, 2.15+0.04j, 2.15+0.01j, -4-4j,
                       6+2j, 6-2j, -6+2j, -6-2j, 0.1+0.3j, 0.1-0.3j, -0.1+0.3j, -0.1-0.3j]

    for k_hat in k_hat_values:
        roots = root_of_func(initial_guesses, k_hat)
        for root in roots:
            omega_hats_imag.append(root.imag)
    return np.max(omega_hats_imag)

# r_ratios の範囲 (1 から 30) を 30 点生成
r_ratios = np.linspace(1, 30, 90)

with Pool() as pool:
    max_imaginary_parts = pool.map(calculate_max_imaginary_part, r_ratios)

plt.figure(figsize=(14, 8))
plt.plot(r_ratios, max_imaginary_parts, lw=2, color='tab:blue',
         label=f"$\\beta(r=1) = {beta_at_1:.3e}$")
plt.xlabel(r"$r/r_{0}$", fontsize=25)
plt.ylabel(r"$\gamma_{max} / \omega_{0}$", fontsize=25)
plt.title("Maximum Growth Rate (MHD) vs r", fontsize=25)
plt.tick_params(axis='both', labelsize=20)
plt.legend(fontsize=16)
plt.grid(True, which='both', ls='--')
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
from io import StringIO
from multiprocessing import Pool

# ======== 既存のデータ読み込みや補間、フィッティングのコード ===========
# （ここでは既に R_target, spline_Tpara, bump_model, popt, beta_para_interp, B_perp_squared_0, beta_interp などが定義されているものとします）

# ----- 例：β の補間関数（MHD計算用） -----
# ※ この beta_interp は、先のコードで作成した R_target, beta のスプライン補間関数です
# 例: beta_interp = UnivariateSpline(R_target, beta, s=0)
# --------------------------------------------------------------------------

# ======== MHD版 最大成長率計算用関数 ========
def calculate_max_imaginary_part_mhd(r_ratio):
    beta_val = beta_interp(r_ratio)  # 例：beta_interp = UnivariateSpline(R_target, beta, s=0)
    B_perp_squared = B_perp_squared_0 * r_ratio**(0.92)

    def newton_method(omega_hat, k_hat):
        f = ((omega_hat - k_hat) * (omega_hat**2 - beta_val * k_hat**2) *
             ((omega_hat + k_hat)**2 - 4) -
             (B_perp_squared * k_hat**2) * (omega_hat**3 + k_hat * omega_hat**2 - 3 * omega_hat + k_hat))
        df = (3 * B_perp_squared * k_hat**2 + beta_val * k_hat**4 + 4 * beta_val * k_hat**2 +
              4 * k_hat * omega_hat**3 + 5 * omega_hat**4 +
              3 * omega_hat**2 * (-B_perp_squared * k_hat**2 - beta_val * k_hat**2 - k_hat**2 - 4) +
              2 * omega_hat * (-B_perp_squared * k_hat**3 - beta_val * k_hat**3 - k_hat**3 + 4 * k_hat))
        if np.abs(df) < 1e-10:
            return omega_hat
        return omega_hat - f / df

    def root_of_func(initial_guesses, k_hat):
        roots = []
        for omega_hat in initial_guesses:
            for _ in range(300):
                omega_hat_new = newton_method(omega_hat, k_hat)
                if np.abs(omega_hat_new - omega_hat) < 1e-10:
                    if all(np.abs(omega_hat_new - root) > 1e-10 for root in roots):
                        roots.append(omega_hat_new)
                    break
                omega_hat = omega_hat_new
        return roots

    k_hat_values = np.linspace(0.85, 2, 1000)
    omega_hats_imag = []
    initial_guesses = [0.1+0.2j, 2+0.5j, -5+2j, -5-8j, 7-3j, 1+1j, 2+0.1j, 2-0.1j,
                       -2+2j, -2-2j, 3+3j, 3-3j, -4+0.5j, 4+0.5j, 0+3j, 0-3j,
                       1+2j, 1-2j, -1+0.5j, -1-0.5j, 0.5+1j, 0.5-1j, -0.5+0.2j,
                       -0.5-0.2j, 3.5+4j, 3.5-4j, -3.5+4j, -3.5-4j,
                       2.15+0.1j, 2.20+0.1j, 2.25+0.1j, 2.1+0.1j, 2.15+0.07j, 2.15+0.04j, 2.15+0.01j, -4-4j,
                       6+2j, 6-2j, -6+2j, -6-2j, 0.1+0.3j, 0.1-0.3j, -0.1+0.3j, -0.1-0.3j]
    for k_hat in k_hat_values:
        roots = root_of_func(initial_guesses, k_hat)
        for root in roots:
            omega_hats_imag.append(root.imag)
    return np.max(omega_hats_imag)

# ======== PDI版 最大成長率計算用関数 ========
def calculate_max_imaginary_part_pdi(r_ratio):
    beta_parallel = beta_para_interp(r_ratio)
    xi = bump_model(r_ratio, *popt)
    B_perp_squared = B_perp_squared_0 * r_ratio**(0.92)
    tilde_beta = beta_parallel * (3.0/2.0) / (1.0 + B_perp_squared + beta_parallel/2.0 * (xi - 1.0))

    def newton_method(omega_hat, k_hat):
        f = ((omega_hat**2 - tilde_beta * k_hat**2 * (1.0 + B_perp_squared * xi / 3.0)) *
             ((omega_hat - k_hat) * ((omega_hat + k_hat)**2 - 4.0) +
              tilde_beta * B_perp_squared * (xi - 4.0) / (3.0*(1.0+B_perp_squared)) *
             ((k_hat**2+1.0)*omega_hat + k_hat*(k_hat**2-3.0))) -
             (B_perp_squared * k_hat**2 * (1.0 - tilde_beta*(3.0-xi-B_perp_squared) / (3.0*(1.0+B_perp_squared))) *
              ((omega_hat**3+omega_hat**2*k_hat-3.0*omega_hat+k_hat) -
               tilde_beta*(3.0-xi)/3.0 * ((k_hat**2+1.0)*omega_hat+k_hat*(k_hat**2-3.0)))))
        df = (2*omega_hat * (((omega_hat-k_hat)*((omega_hat+k_hat)**2-4.0)) +
              tilde_beta*B_perp_squared*(xi-4.0)/(3.0*(1.0+B_perp_squared)) *
              ((k_hat**2+1.0)*omega_hat+k_hat*(k_hat**2-3.0))) +
              (omega_hat**2 - tilde_beta*k_hat**2*(1.0+B_perp_squared*xi/3.0)) *
              (((omega_hat+k_hat)**2-4.0) + (omega_hat-k_hat)*2*(omega_hat+k_hat)) +
              tilde_beta*B_perp_squared*(xi-4.0)/(3.0*(1.0+B_perp_squared))*(k_hat**2+1.0) -
              B_perp_squared*k_hat**2*(1.0 - tilde_beta*(3.0-xi-B_perp_squared) / (3.0*(1.0+B_perp_squared))) *
              (3*omega_hat**2+2*omega_hat*k_hat-3.0) -
              (B_perp_squared*k_hat**2 * (-tilde_beta*(3.0-xi)/3.0))*(k_hat**2+1.0)
             )
        if np.abs(df) < 1e-10:
            return omega_hat
        return omega_hat - f/df

    def root_of_func(initial_guesses, k_hat):
        roots = []
        for omega_hat in initial_guesses:
            for _ in range(1000):
                omega_hat_new = newton_method(omega_hat, k_hat)
                if np.abs(omega_hat_new - omega_hat) < 1e-10:
                    if all(np.abs(omega_hat_new - root) > 1e-10 for root in roots):
                        roots.append(omega_hat_new)
                    break
                omega_hat = omega_hat_new
        return roots

    k_hat_values = np.linspace(0.85, 2, 1000)
    omega_hats_imag = []
    initial_guesses = [0.1+0.2j, 2+0.5j, -5+2j, -5-8j, 7-3j, 1+1j, 2+0.1j, 2-0.1j,
                       -2+2j, -2-2j, 3+3j, 3-3j, -4+0.5j, 4+0.5j, 0+3j, 0-3j,
                       1+2j, 1-2j, -1+0.5j, -1-2j, 0.5+1j, 0.5-1j, -0.5+0.2j,
                       -0.5-0.2j, 3.5+4j, 3.5-4j, -3.5+4j, -3.5-4j,
                       2.15+0.1j, 2.20+0.1j, 2.25+0.1j, 2.1+0.1j, 2.15+0.07j, 2.15+0.04j, 2.15+0.01j, -4-4j,
                       6+2j, 6-2j, -6+2j, -6-2j, 0.1+0.3j, 0.1-0.3j, -0.1+0.3j, -0.1-0.3j]
    for k_hat in k_hat_values:
        roots = root_of_func(initial_guesses, k_hat)
        for root in roots:
            omega_hats_imag.append(root.imag)
    return np.max(omega_hats_imag)

# ======== 両者の計算と同一Figureでのプロット ============
# 共通の r_ratio の範囲（例：1～30）
r_ratios = np.linspace(1, 30, 90)

# MHD版最大成長率の計算
with Pool() as pool:
    max_imag_mhd = pool.map(calculate_max_imaginary_part_mhd, r_ratios)

# PDI版の場合は Firehose 条件により計算可能な範囲を制限する（例）
def calculate_firehose_condition(r):
    beta_parallel = beta_para_interp(r)
    xi = bump_model(r, *popt)
    B_perp_squared = B_perp_squared_0 * r**(0.92)
    return 1 + 0.5 * beta_parallel * (xi - 1) / (1 + B_perp_squared)

firehose_values = np.array([calculate_firehose_condition(r) for r in r_ratios])
cutoff_index = np.argmax(firehose_values < 0) if np.any(firehose_values < 0) else len(r_ratios)
r_ratios_pdi = r_ratios[:cutoff_index]

with Pool() as pool:
    max_imag_pdi = pool.map(calculate_max_imaginary_part_pdi, r_ratios_pdi)

# プロット
plt.rcParams["font.size"] = 20
plt.figure(figsize=(14, 8))
plt.plot(r_ratios, max_imag_mhd, lw=2, color='tab:blue', label=r"$\gamma_{max}/\omega_{0} (isotropic)$")
plt.plot(r_ratios_pdi, max_imag_pdi, lw=2, color='tab:red', label=r"$\gamma_{max}/\omega_{0}$ (CGL)")
plt.xlabel(r"$R/R_{0}$", fontsize=25)
plt.ylabel(r"$\gamma_{max}/\omega_{0}$", fontsize=25)
plt.title("$\gamma_{max}/\omega_{0}$ vs r", fontsize=25)
plt.legend(fontsize=18)
plt.grid(True)
plt.tight_layout()
plt.show()

# ▼ 追加セル ──────────────────────────────────────────────
# ---------------------------------------------------------
# 1) 物理パラメータを R 方向に並べた配列を準備
# ---------------------------------------------------------
beta_iso_vals       = beta_interp(r_ratios)              # 等方 MHD の β
beta_parallel_vals  = beta_para_interp(r_ratios)         # CGL β||
xi_vals             = bump_model(r_ratios, *popt)        # CGL ξ
B_perp_vals         = B_perp_squared_0 * r_ratios**(0.92)     # B⊥² (r¹ スケーリング)

# ---------------------------------------------------------
# 2) 連結プロット作成
# ---------------------------------------------------------
fig, axes = plt.subplots(
    nrows=2, ncols=1, sharex=True, figsize=(14, 10),
    gridspec_kw={'hspace': 0.04, 'height_ratios': [1.5, 1]}
)

# ── (i) 上段：成長率比較 ──────────────────────────────
ax_top = axes[0]
ax_top.plot(r_ratios,      max_imag_mhd, lw=2, color='tab:blue',
            label=r'$\gamma_{\max}/\omega_{0}\;$(isotropic)')
ax_top.plot(r_ratios_pdi,  max_imag_pdi, lw=2, color='tab:red',
            label=r'$\gamma_{\max}/\omega_{0}\;$(CGL)')

ax_top.set_ylabel(r'$\gamma_{\max}/\omega_{0}$')
ax_top.grid(True, alpha=0.3)
ax_top.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

# ── (ii) 下段：β, β∥, ξ, 𝐵⊥² ─────────────────────────
ax_bot = axes[1]

ax_bot.plot(r_ratios, beta_iso_vals,      lw=2, color='tab:brown',
            label=r'$\beta$ (isotropic)')
ax_bot.plot(r_ratios, beta_parallel_vals, lw=2, color='tab:purple',
            label=r'$\beta_{\parallel}$ (CGL)')
ax_bot.plot(r_ratios, xi_vals,            lw=2, color='tab:orange',
            label=r'$\xi=T_{\perp}/T_{\parallel}$')
ax_bot.plot(r_ratios, B_perp_vals,        lw=2, color='tab:cyan',
            label=r'$\hat{B}_{\perp}^{2}$')

ax_bot.set_yscale('log')
ax_bot.set_xlabel(r'$R/R_{0}$')
ax_bot.set_ylabel(r'$\beta,\;\beta_{\parallel},\;\xi,\;\hat{B}_{\perp}^{2}$')
ax_bot.set_ylim(1e-5, 1e2)
ax_bot.set_yticks([1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2])
from matplotlib.ticker import LogLocator
# ───── log‐grid 追加ここから ─────
ax_bot.yaxis.set_minor_locator(LogLocator(base=10.0,          # 10 のべき
                                       subs=np.arange(1, 10)*0.1,  # 1–9 の副目盛り
                                       numticks=100))
ax_bot.grid(True,  which='major', alpha=0.6)  # メジャー
ax_bot.grid(True,  which='minor', axis='y', alpha=0.3)  # マイナー
# ───── ここまで ─────
ax_bot.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

# ── レイアウト調整 ──────────────────────────────────
plt.subplots_adjust(right=0.8)
plt.tight_layout()
plt.show()
# ▲ 追加セル ──────────────────────────────────────────────

# ▼▼▼ ここから追記 ─────────────────────────────────────────
# -----------------------------------------------------------
# 6. データセット（90 × 7）を作成して CSV に保存
#     0: r_ratio
#     1: γ_max/ω0  (CGL - PDI)           … Firehoseが負になる r では NaN
#     2: γ_max/ω0  (isotropic-MHD)
#     3: β         (isotropic-MHD)
#     4: β_parallel(CGL - PDI)
#     5: ξ         (=T_perp/T_parallel)
#     6: B_perp²   (≡ B_perp_squared_0 · r^0.92)
# -----------------------------------------------------------

# ---- ① CGL 成長率を全 r 90 点に並べる（Firehoseカット以降は NaN） ----
gamma_CGL_full = np.full_like(r_ratios, np.nan, dtype=float)
gamma_CGL_full[:cutoff_index] = max_imag_pdi   # PDI 計算で得た値を先頭から代入

# ---- ② 列方向にスタックしてデータセット完成 ----------------------------
dataset = np.column_stack([
    r_ratios,           # 0
    gamma_CGL_full,     # 1
    max_imag_mhd,       # 2
    beta_iso_vals,      # 3
    beta_parallel_vals, # 4
    xi_vals,            # 5
    B_perp_vals         # 6
])

# ---- ③ CSV 出力 ---------------------------------------------------------
header = ("R_over_R0, gamma_CGL, gamma_isotropic, "
          "beta_isotropic, beta_parallel, xi, B_perp_squared")

np.savetxt(
    "Fig4c_dataset_full.csv",
    dataset,
    delimiter=",",
    header=header,
    comments=""
)

print(">> Fig4c_dataset_full.csv を保存しました")
print("\n--- preview ---")
print(dataset[:5])   # 先頭5行を確認
# ▲▲▲ ここまで追記 ─────────────────────────────────────────


















# シナリオ３　平行方向等温で垂直方向断熱、磁場と密度はPSPスケーリング　beta=0.001　本当の完成版

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import matplotlib.ticker as mticker

# ===== ここから追加／置き換え ===================================
plt.rcParams.update({
    'font.size': 20,        # デフォルトのフォントサイズ
    'axes.titlesize': 20,   # タイトル
    'axes.labelsize': 20,   # 軸ラベル
    'xtick.labelsize': 20,  # x 目盛ラベル
    'ytick.labelsize': 20,  # y 目盛ラベル
    'legend.fontsize': 20   # 凡例
})
# ================================================================

flg = 18

a = 0.73
b = -1.66
c = 0.92

# 固定定数
B_perp_squared_0 = 0.001  # 初期 B_perp^2 の値

# CGLモデル用の初期値
beta_parallel_0 = 0.001/7  # 初期 beta_parallel_0
xi_0 = 10             # 初期 xi_0

# isotropicモデル用の定数（CGLとは別に使う）
n0 = 1e8              # 単位: cm^-3 (例)
k_b = 1.38e-16        # erg/K (例)
B0 = 50               # 初期磁場（任意単位）
#beta_0 = 0.001        # isotropic用基準 beta_0

# ------------------------------
# CGLモデルのγmax/ω0の計算関数
def calculate_max_imaginary_part_CGL(r_ratio, xi_0, beta_parallel_0):
    beta_parallel = beta_parallel_0 * r_ratio**(a)
    xi = xi_0 * r_ratio**(b)
    B_perp_squared = B_perp_squared_0 * r_ratio**(c)
    tilde_beta = beta_parallel * (3.0/2.0) / (1.0 + B_perp_squared + beta_parallel/2.0*(xi - 1.0))

    def newton_method(omega_hat, k_hat):
        f = ((omega_hat**2 - tilde_beta * k_hat**2 * (1.0 + B_perp_squared * xi / 3.0)) *
             ((omega_hat - k_hat) * ((omega_hat + k_hat)**2 - 4.0) +
              tilde_beta * B_perp_squared * (xi - 4.0) / (3.0*(1.0 + B_perp_squared)) *
              ((k_hat**2+1.0)*omega_hat + k_hat*(k_hat**2-3.0))) -
             (B_perp_squared * k_hat**2 * (1.0 - tilde_beta*(3.0 - xi - B_perp_squared) / (3.0*(1.0+B_perp_squared))) *
              ((omega_hat**3+omega_hat**2*k_hat-3.0*omega_hat+k_hat) - tilde_beta*(3.0 - xi)/3.0 *
               ((k_hat**2+1.0)*omega_hat + k_hat*(k_hat**2-3.0)))))
        df = (2 * omega_hat * ((omega_hat - k_hat)*((omega_hat+k_hat)**2 - 4.0) +
                tilde_beta*B_perp_squared*(xi-4.0)/(3.0*(1.0+B_perp_squared))*((k_hat**2+1.0)*omega_hat+k_hat*(k_hat**2-3.0))) +
              (omega_hat**2 - tilde_beta*k_hat**2*(1.0+B_perp_squared*xi/3.0))*
              (((omega_hat+k_hat)**2-4.0)+(omega_hat-k_hat)*2*(omega_hat+k_hat)) +
              tilde_beta*B_perp_squared*(xi-4.0)/(3.0*(1.0+B_perp_squared))*(k_hat**2+1.0) -
              B_perp_squared*k_hat**2*(1.0-tilde_beta*(3.0-xi-B_perp_squared)/(3.0*(1.0+B_perp_squared)))*
              (3*omega_hat**2+2*omega_hat*k_hat-3.0) -
              (B_perp_squared*k_hat**2*(-tilde_beta*(3.0-xi)/3.0))*(k_hat**2+1.0) )
        if np.abs(df) < 1e-10:
            return omega_hat
        return omega_hat - f/df

    def root_of_func(initial_guesses, k_hat):
        roots = []
        for omega_hat in initial_guesses:
            for _ in range(300):
                omega_hat_new = newton_method(omega_hat, k_hat)
                if np.abs(omega_hat_new - omega_hat) < 1e-10:
                    if all(np.abs(omega_hat_new - root) > 1e-10 for root in roots):
                        roots.append(omega_hat_new)
                    break
                omega_hat = omega_hat_new
        return roots

    k_hat_values = np.linspace(0.85, 2.5, 1)
    initial_guesses = [0.1+0.2j, 2+0.5j, -5+2j, -5-8j, 7-3j, 1+1j, 2+0.1j, 2-0.1j,
                       -2+2j, -2-2j, 3+3j, 3-3j, -4+0.5j, 4+0.5j, 0+3j, 0-3j,
                       1+2j, 1-2j, -1+0.5j, -1-0.5j, 0.5+1j, 0.5-1j, -0.5+0.2j,
                       -0.5-0.2j, 3.5+4j, 3.5-4j, -3.5+4j, -3.5-4j,
                       2.15+0.1j, 2.20+0.1j, 2.25+0.1j, 2.1+0.1j, 2.15+0.07j, 2.15+0.04j, 2.15+0.01j, -4-4j,
                       6+2j, 6-2j, -6+2j, -6-2j, 0.1+0.3j, 0.1-0.3j, -0.1+0.3j, -0.1-0.3j]
    omega_hats_imag = []
    for k_hat in k_hat_values:
        roots = root_of_func(initial_guesses, k_hat)
        for root in roots:
            omega_hats_imag.append(root.imag)
    return np.max(omega_hats_imag)

# ------------------------------
# isotropicモデルのγmax/ω0の計算関数
def calculate_max_imaginary_part_isotropic(args):
    r_ratio, B_perp_squared_0 = args
    # 新たな β の計算（density, B, T のスケーリングに基づく）
    n = n0 * r_ratio**(-2.59)
    B = B0 * r_ratio**(-1.66)
    T = 1.03e6 * (1 + 20 * r_ratio**(-1.66)) / 3
    beta = n * k_b * T * 8 * np.pi / (B**2)

    B_perp_squared = B_perp_squared_0 * r_ratio**(c)

    def newton_method(omega_hat, k_hat):
        f = (omega_hat - k_hat) * (omega_hat**2 - beta*k_hat**2) * ((omega_hat+k_hat)**2 - 4) \
            - (B_perp_squared * k_hat**2) * (omega_hat**3 + k_hat*omega_hat**2 - 3*omega_hat + k_hat)
        df = (3 * B_perp_squared * k_hat**2 + beta*k_hat**4 + 4*beta*k_hat**2 +
              4*k_hat*omega_hat**3 + 5*omega_hat**4 +
              3*omega_hat**2 * (-B_perp_squared*k_hat**2 - beta*k_hat**2 - k_hat**2 - 4) +
              2*omega_hat * (-B_perp_squared*k_hat**3 - beta*k_hat**3 - k_hat**3 + 4*k_hat))
        if np.abs(df) < 1e-10:
            return omega_hat
        return omega_hat - f/df

    def root_of_func(initial_guesses, k_hat):
        roots = []
        for omega_hat in initial_guesses:
            for _ in range(300):
                omega_hat_new = newton_method(omega_hat, k_hat)
                if np.abs(omega_hat_new - omega_hat) < 1e-10:
                    if all(np.abs(omega_hat_new - root) > 1e-10 for root in roots):
                        roots.append(omega_hat_new)
                    break
                omega_hat = omega_hat_new
        return roots

    k_hat_values = np.linspace(0.85, 2.5, 1)
    initial_guesses = [0.1+0.2j, 2+0.5j, -5+2j, -5-8j, 7-3j, 1+1j, 2+0.1j, 2-0.1j,
                       -2+2j, -2-2j, 3+3j, 3-3j, -4+0.5j, 4+0.5j, 0+3j, 0-3j,
                       1+2j, 1-2j, -1+0.5j, -1-0.5j, 0.5+1j, 0.5-1j, -0.5+0.2j,
                       -0.5-0.2j, 3.5+4j, 3.5-4j, -3.5+4j, -3.5-4j,
                       2.15+0.1j, 2.20+0.1j, 2.25+0.1j, 2.1+0.1j, 2.15+0.07j, 2.15+0.04j, 2.15+0.01j, -4-4j,
                       6+2j, 6-2j, -6+2j, -6-2j, 0.1+0.3j, 0.1-0.3j, -0.1+0.3j, -0.1-0.3j]
    omega_hats_imag = []
    for k_hat in k_hat_values:
        roots = root_of_func(initial_guesses, k_hat)
        for root in roots:
            omega_hats_imag.append(root.imag)
    return np.max(omega_hats_imag)

# ------------------------------
# Firehose条件（CGLモデル用）の計算関数
def calculate_firehose_condition(r_ratios, xi_0, beta_parallel_0, B_perp_squared_0):
    # xi, beta_parallel, B_perp_squared のスケーリング（CGLモデル）
    xi_values = xi_0 * r_ratios**(b)
    beta_parallel_values = beta_parallel_0 * r_ratios**(a)
    B_perp_squared_values = B_perp_squared_0 * r_ratios**(c)
    condition = 1 + 0.5 * beta_parallel_values * (xi_values - 1) / (1 + B_perp_squared_values)
    return condition

# ------------------------------
# r_ratio の範囲
r_ratios = np.linspace(1, 30, 90)

# (A) CGLモデルのγmax/ω0の計算（Firehose条件が正となる領域のみ）
firehose_condition = calculate_firehose_condition(r_ratios, xi_0, beta_parallel_0, B_perp_squared_0)
cutoff_index = np.argmax(firehose_condition < 0) if np.any(firehose_condition < 0) else len(r_ratios)
r_ratios_cutoff = r_ratios[:cutoff_index]

with Pool() as pool:
    max_imaginary_parts_CGL = pool.starmap(
        calculate_max_imaginary_part_CGL,
        [(r, xi_0, beta_parallel_0) for r in r_ratios_cutoff]
    )
    max_imaginary_parts_CGL = [result for result in max_imaginary_parts_CGL]

# (B) isotropicモデルのγmax/ω0の計算（全r_ratio）
with Pool() as pool:
    max_imaginary_parts_iso = pool.map(
        calculate_max_imaginary_part_isotropic,
        [(r, B_perp_squared_0) for r in r_ratios]
    )
    max_imaginary_parts_iso = [result for result in max_imaginary_parts_iso]

# =========================================================
# 連結プロット：上段を下段より少し大きく
# =========================================================
fig, axes = plt.subplots(
    nrows=2, ncols=1, sharex=True, figsize=(15, 10),
    gridspec_kw={
        'hspace': 0.05,           # パネル間の縦スペース
        'height_ratios': [1.7, 1] # ← ここで比率を指定（[上段, 下段]）
    })

# -------------------------
# (1) 上段: γmax / ω0 & Firehose
# -------------------------
ax1  = axes[0]
ax1_r = ax1.twinx()

ax1.plot(r_ratios_cutoff, max_imaginary_parts_CGL,
         color='tab:red', lw=2, label=r"$\gamma_{max}/\omega_{0}$ (CGL)")
ax1.plot(r_ratios, max_imaginary_parts_iso,
         color='tab:green', lw=2, label=r"$\gamma_{max}/\omega_{0}$ (isotropic)")
ax1_r.plot(r_ratios, firehose_condition,
           color='tab:blue', lw=2, label="Discriminant value")

ax1.set_ylabel(r"$\gamma_{max}/\omega_{0}$")
ax1.tick_params(axis='y')
ax1.grid(True, alpha=0.3)
ax1_r.set_ylabel("Discriminant value", color='tab:blue')
ax1_r.tick_params(axis='y',colors='tab:blue')
ax1_r.set_ylim(-1, 1.5)

# 上段凡例（右外）
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_r.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,
           loc='center left', bbox_to_anchor=(1.20, 0.5),
           frameon=False)

# -------------------------
# (2) 下段: β∥, β, ξ, B⊥²
# -------------------------
ax2 = axes[1]

n = n0 * r_ratios**(-2.59)
B = B0 * r_ratios**(-1.66)
T = 1.03e6 * (1 + 20 * r_ratios**(-1.66)) / 3

beta_parallel_vals = beta_parallel_0 * r_ratios**(a)
beta_iso_vals = n * k_b * T * 8 * np.pi / (B**2)
xi_vals            = xi_0 * r_ratios**(b)
B_perp_vals        = B_perp_squared_0 * r_ratios**(c)

ax2.plot(r_ratios, beta_parallel_vals, color='tab:purple', lw=2,
         label=r'$\beta_{\parallel}$ (CGL)')
ax2.plot(r_ratios, beta_iso_vals,      color='tab:brown',  lw=2,
         label=r'$\beta$ (isotropic)')
ax2.plot(r_ratios, xi_vals,            color='tab:orange', lw=2,
         label=r'$\xi = T_\perp/T_\parallel$')
ax2.plot(r_ratios, B_perp_vals,        color='tab:cyan',   lw=2,
         label=r'$\hat{B}_\perp^{2}$')

ax2.set_yscale('log')
ax2.set_xlabel(r'$R/R_{0}$')
ax2.set_ylabel(r'$\beta_{\parallel},\; \beta,\; \xi,\; \hat{B}_\perp^{2}$')
#ax2.set_ylim(0.00001,100)
ax2.set_yticks([10e-5, 1e-4, 1e-3, 1e-2,
                1e-1, 1e0, 1e1,1e2])
ax2.grid(True, which='both', ls='--', alpha=0.3)
# 10^x 表記に（指数を上付きで）
ax2.get_yaxis().set_major_formatter(
    mticker.LogFormatterMathtext(base=10)
)

# ========= ここを置き換え =========
# 既存の 1 行 grid は削除して、下の 4 行に差し替え
ax2.grid(True, which='both', ls='--', alpha=0.3)   # 主グリッド
ax2.set_axisbelow(True)                # 曲線よりも背面にグリッドを描く
# ==================================

# 下段凡例（右外）
ax2.legend(loc='center left', bbox_to_anchor=(1.20, 0.5),
           frameon=False)

# -------------------------
# 図全体レイアウト
# -------------------------
plt.subplots_adjust(right=0.8)  # 右 20 % を凡例スペースに
plt.show()

# ===============================
# 4. 各 r_ratio に対するデータセット (90×3) の作成と出力
# ===============================
# 出力するデータセットの各行は [max_imaginary_CGL, max_imaginary_isotropic, firehose_condition] となる
# ※ CGL の値は、firehose 条件が正の範囲 (r < r_ratios_cutoff) では計算済み。条件が成立しない r_ratio には NaN を補完

# 90 行のデータセットを作成
dataset = np.empty((len(r_ratios), 3))
dataset[:, 1] = max_imaginary_parts_iso          # 等方モデルの値は全 r_ratio に対してある
dataset[:, 2] = firehose_condition               # Firehose 条件

# CGL の値は、cutoff_index まで計算されているので、それ以降は NaN とする
dataset[:cutoff_index, 0] = max_imaginary_parts_CGL
dataset[cutoff_index:, 0] = np.nan

# dataset の各行に対応する r_ratio の値も表示したい場合は、別途 r_ratios を利用するか、カラムとして結合する
# 例: [r_ratio, max_imaginary_CGL, max_imaginary_isotropic, firehose_condition] の 90×4 のデータセット
# 今回は 90×3 (各モデルの値のみ) を出力する

# データセットの出力（例：標準出力に表示）
print("行: r_ratio のインデックス (1～90)")
print("列: [γₘₐₓ/ω₀ (CGL), γₘₐₓ/ω₀ (isotropic), Firehose 条件]")
print(dataset)

# -------------------------
# 追加: 下段プロット用データも含めて保存
# -------------------------
# 8 列: r, γCGL, γiso, Firehose, β∥, β, ξ, B⊥²
full_dataset = np.column_stack([
    r_ratios,
    dataset,                   # 既存の 3 列
    beta_parallel_vals,
    beta_iso_vals,
    xi_vals,
    B_perp_vals
])

# ヘッダー行を列名に合わせて作成
header = ("r_ratio, gamma_CGL, gamma_iso, firehose, "
          "beta_parallel, beta_iso, xi, B_perp_squared")

# 標準出力で確認
print("\n--- full_dataset preview ---")
print(full_dataset[:5])   # 先頭 5 行だけ表示

# CSV 保存
np.savetxt("Fig3a_dataset_full.csv", full_dataset,
           delimiter=",", header=header, comments="")
print(">> Fig3a_dataset_full.csv を保存しました")













#下段の図をlogスケールグリッドでプロットするための実験コード  
#シナリオ３　平行方向等温で垂直方向断熱、磁場と密度はPSPスケーリング　beta=0.001　本当の完成版

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import matplotlib.ticker as mticker

# ===== ここから追加／置き換え ===================================
plt.rcParams.update({
    'font.size': 20,        # デフォルトのフォントサイズ
    'axes.titlesize': 20,   # タイトル
    'axes.labelsize': 20,   # 軸ラベル
    'xtick.labelsize': 20,  # x 目盛ラベル
    'ytick.labelsize': 20,  # y 目盛ラベル
    'legend.fontsize': 20   # 凡例
})
# ================================================================

flg = 18

a = 0.73
b = -1.66
c = 0.92

# 固定定数
B_perp_squared_0 = 0.001  # 初期 B_perp^2 の値

# CGLモデル用の初期値
beta_parallel_0 = 0.001/7  # 初期 beta_parallel_0
xi_0 = 10             # 初期 xi_0

# isotropicモデル用の定数（CGLとは別に使う）
n0 = 1e8              # 単位: cm^-3 (例)
k_b = 1.38e-16        # erg/K (例)
B0 = 50               # 初期磁場（任意単位）
#beta_0 = 0.001        # isotropic用基準 beta_0

# ------------------------------
# CGLモデルのγmax/ω0の計算関数
def calculate_max_imaginary_part_CGL(r_ratio, xi_0, beta_parallel_0):
    beta_parallel = beta_parallel_0 * r_ratio**(a)
    xi = xi_0 * r_ratio**(b)
    B_perp_squared = B_perp_squared_0 * r_ratio**(c)
    tilde_beta = beta_parallel * (3.0/2.0) / (1.0 + B_perp_squared + beta_parallel/2.0*(xi - 1.0))

    def newton_method(omega_hat, k_hat):
        f = ((omega_hat**2 - tilde_beta * k_hat**2 * (1.0 + B_perp_squared * xi / 3.0)) *
             ((omega_hat - k_hat) * ((omega_hat + k_hat)**2 - 4.0) +
              tilde_beta * B_perp_squared * (xi - 4.0) / (3.0*(1.0 + B_perp_squared)) *
              ((k_hat**2+1.0)*omega_hat + k_hat*(k_hat**2-3.0))) -
             (B_perp_squared * k_hat**2 * (1.0 - tilde_beta*(3.0 - xi - B_perp_squared) / (3.0*(1.0+B_perp_squared))) *
              ((omega_hat**3+omega_hat**2*k_hat-3.0*omega_hat+k_hat) - tilde_beta*(3.0 - xi)/3.0 *
               ((k_hat**2+1.0)*omega_hat + k_hat*(k_hat**2-3.0)))))
        df = (2 * omega_hat * ((omega_hat - k_hat)*((omega_hat+k_hat)**2 - 4.0) +
                tilde_beta*B_perp_squared*(xi-4.0)/(3.0*(1.0+B_perp_squared))*((k_hat**2+1.0)*omega_hat+k_hat*(k_hat**2-3.0))) +
              (omega_hat**2 - tilde_beta*k_hat**2*(1.0+B_perp_squared*xi/3.0))*
              (((omega_hat+k_hat)**2-4.0)+(omega_hat-k_hat)*2*(omega_hat+k_hat)) +
              tilde_beta*B_perp_squared*(xi-4.0)/(3.0*(1.0+B_perp_squared))*(k_hat**2+1.0) -
              B_perp_squared*k_hat**2*(1.0-tilde_beta*(3.0-xi-B_perp_squared)/(3.0*(1.0+B_perp_squared)))*
              (3*omega_hat**2+2*omega_hat*k_hat-3.0) -
              (B_perp_squared*k_hat**2*(-tilde_beta*(3.0-xi)/3.0))*(k_hat**2+1.0) )
        if np.abs(df) < 1e-10:
            return omega_hat
        return omega_hat - f/df

    def root_of_func(initial_guesses, k_hat):
        roots = []
        for omega_hat in initial_guesses:
            for _ in range(300):
                omega_hat_new = newton_method(omega_hat, k_hat)
                if np.abs(omega_hat_new - omega_hat) < 1e-10:
                    if all(np.abs(omega_hat_new - root) > 1e-10 for root in roots):
                        roots.append(omega_hat_new)
                    break
                omega_hat = omega_hat_new
        return roots

    k_hat_values = np.linspace(0.85, 2.5, 1)
    initial_guesses = [0.1+0.2j, 2+0.5j, -5+2j, -5-8j, 7-3j, 1+1j, 2+0.1j, 2-0.1j,
                       -2+2j, -2-2j, 3+3j, 3-3j, -4+0.5j, 4+0.5j, 0+3j, 0-3j,
                       1+2j, 1-2j, -1+0.5j, -1-0.5j, 0.5+1j, 0.5-1j, -0.5+0.2j,
                       -0.5-0.2j, 3.5+4j, 3.5-4j, -3.5+4j, -3.5-4j,
                       2.15+0.1j, 2.20+0.1j, 2.25+0.1j, 2.1+0.1j, 2.15+0.07j, 2.15+0.04j, 2.15+0.01j, -4-4j,
                       6+2j, 6-2j, -6+2j, -6-2j, 0.1+0.3j, 0.1-0.3j, -0.1+0.3j, -0.1-0.3j]
    omega_hats_imag = []
    for k_hat in k_hat_values:
        roots = root_of_func(initial_guesses, k_hat)
        for root in roots:
            omega_hats_imag.append(root.imag)
    return np.max(omega_hats_imag)

# ------------------------------
# isotropicモデルのγmax/ω0の計算関数
def calculate_max_imaginary_part_isotropic(args):
    r_ratio, B_perp_squared_0 = args
    # 新たな β の計算（density, B, T のスケーリングに基づく）
    n = n0 * r_ratio**(-2.59)
    B = B0 * r_ratio**(-1.66)
    T = 1.03e6 * (1 + 20 * r_ratio**(-1.66)) / 3
    beta = n * k_b * T * 8 * np.pi / (B**2)

    B_perp_squared = B_perp_squared_0 * r_ratio**(c)

    def newton_method(omega_hat, k_hat):
        f = (omega_hat - k_hat) * (omega_hat**2 - beta*k_hat**2) * ((omega_hat+k_hat)**2 - 4) \
            - (B_perp_squared * k_hat**2) * (omega_hat**3 + k_hat*omega_hat**2 - 3*omega_hat + k_hat)
        df = (3 * B_perp_squared * k_hat**2 + beta*k_hat**4 + 4*beta*k_hat**2 +
              4*k_hat*omega_hat**3 + 5*omega_hat**4 +
              3*omega_hat**2 * (-B_perp_squared*k_hat**2 - beta*k_hat**2 - k_hat**2 - 4) +
              2*omega_hat * (-B_perp_squared*k_hat**3 - beta*k_hat**3 - k_hat**3 + 4*k_hat))
        if np.abs(df) < 1e-10:
            return omega_hat
        return omega_hat - f/df

    def root_of_func(initial_guesses, k_hat):
        roots = []
        for omega_hat in initial_guesses:
            for _ in range(300):
                omega_hat_new = newton_method(omega_hat, k_hat)
                if np.abs(omega_hat_new - omega_hat) < 1e-10:
                    if all(np.abs(omega_hat_new - root) > 1e-10 for root in roots):
                        roots.append(omega_hat_new)
                    break
                omega_hat = omega_hat_new
        return roots

    k_hat_values = np.linspace(0.85, 2.5, 1)
    initial_guesses = [0.1+0.2j, 2+0.5j, -5+2j, -5-8j, 7-3j, 1+1j, 2+0.1j, 2-0.1j,
                       -2+2j, -2-2j, 3+3j, 3-3j, -4+0.5j, 4+0.5j, 0+3j, 0-3j,
                       1+2j, 1-2j, -1+0.5j, -1-0.5j, 0.5+1j, 0.5-1j, -0.5+0.2j,
                       -0.5-0.2j, 3.5+4j, 3.5-4j, -3.5+4j, -3.5-4j,
                       2.15+0.1j, 2.20+0.1j, 2.25+0.1j, 2.1+0.1j, 2.15+0.07j, 2.15+0.04j, 2.15+0.01j, -4-4j,
                       6+2j, 6-2j, -6+2j, -6-2j, 0.1+0.3j, 0.1-0.3j, -0.1+0.3j, -0.1-0.3j]
    omega_hats_imag = []
    for k_hat in k_hat_values:
        roots = root_of_func(initial_guesses, k_hat)
        for root in roots:
            omega_hats_imag.append(root.imag)
    return np.max(omega_hats_imag)

# ------------------------------
# Firehose条件（CGLモデル用）の計算関数
def calculate_firehose_condition(r_ratios, xi_0, beta_parallel_0, B_perp_squared_0):
    # xi, beta_parallel, B_perp_squared のスケーリング（CGLモデル）
    xi_values = xi_0 * r_ratios**(b)
    beta_parallel_values = beta_parallel_0 * r_ratios**(a)
    B_perp_squared_values = B_perp_squared_0 * r_ratios**(c)
    condition = 1 + 0.5 * beta_parallel_values * (xi_values - 1) / (1 + B_perp_squared_values)
    return condition

# ------------------------------
# r_ratio の範囲
r_ratios = np.linspace(1, 30, 90)

# (A) CGLモデルのγmax/ω0の計算（Firehose条件が正となる領域のみ）
firehose_condition = calculate_firehose_condition(r_ratios, xi_0, beta_parallel_0, B_perp_squared_0)
cutoff_index = np.argmax(firehose_condition < 0) if np.any(firehose_condition < 0) else len(r_ratios)
r_ratios_cutoff = r_ratios[:cutoff_index]

with Pool() as pool:
    max_imaginary_parts_CGL = pool.starmap(
        calculate_max_imaginary_part_CGL,
        [(r, xi_0, beta_parallel_0) for r in r_ratios_cutoff]
    )
    max_imaginary_parts_CGL = [result for result in max_imaginary_parts_CGL]

# (B) isotropicモデルのγmax/ω0の計算（全r_ratio）
with Pool() as pool:
    max_imaginary_parts_iso = pool.map(
        calculate_max_imaginary_part_isotropic,
        [(r, B_perp_squared_0) for r in r_ratios]
    )
    max_imaginary_parts_iso = [result for result in max_imaginary_parts_iso]

# =========================================================
# 連結プロット：上段を下段より少し大きく
# =========================================================
fig, axes = plt.subplots(
    nrows=2, ncols=1, sharex=True, figsize=(15, 10),
    gridspec_kw={
        'hspace': 0.05,           # パネル間の縦スペース
        'height_ratios': [1.7, 1] # ← ここで比率を指定（[上段, 下段]）
    })

# -------------------------
# (1) 上段: γmax / ω0 & Firehose
# -------------------------
ax1  = axes[0]
ax1_r = ax1.twinx()

ax1.plot(r_ratios_cutoff, max_imaginary_parts_CGL,
         color='tab:red', lw=2, label=r"$\gamma_{max}/\omega_{0}$ (CGL)")
ax1.plot(r_ratios, max_imaginary_parts_iso,
         color='tab:green', lw=2, label=r"$\gamma_{max}/\omega_{0}$ (isotropic)")
ax1_r.plot(r_ratios, firehose_condition,
           color='tab:blue', lw=2, label="Discriminant value")

ax1.set_ylabel(r"$\gamma_{max}/\omega_{0}$")
ax1.tick_params(axis='y')
ax1.grid(True, alpha=0.3)
ax1_r.set_ylabel("Discriminant value", color='tab:blue')
ax1_r.tick_params(axis='y',colors='tab:blue')
ax1_r.set_ylim(-1, 1.5)

# 上段凡例（右外）
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_r.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,
           loc='center left', bbox_to_anchor=(1.20, 0.5),
           frameon=False)

# -------------------------
# (2) 下段: β∥, β, ξ, B⊥²
# -------------------------
ax2 = axes[1]

n = n0 * r_ratios**(-2.59)
B = B0 * r_ratios**(-1.66)
T = 1.03e6 * (1 + 20 * r_ratios**(-1.66)) / 3

beta_parallel_vals = beta_parallel_0 * r_ratios**(a)
beta_iso_vals = n * k_b * T * 8 * np.pi / (B**2)
xi_vals            = xi_0 * r_ratios**(b)
B_perp_vals        = B_perp_squared_0 * r_ratios**(c)

ax2.plot(r_ratios, beta_iso_vals,      lw=2, color='tab:brown',
            label=r'$\beta$ (isotropic)')
ax2.plot(r_ratios, beta_parallel_vals, lw=2, color='tab:purple',
            label=r'$\beta_{\parallel}$ (CGL)')
ax2.plot(r_ratios, xi_vals,            lw=2, color='tab:orange',
            label=r'$\xi=T_{\perp}/T_{\parallel}$')
ax2.plot(r_ratios, B_perp_vals,        lw=2, color='tab:cyan',
            label=r'$\hat{B}_{\perp}^{2}$')

ax2.set_yscale('log')
ax2.set_xlabel(r'$R/R_{0}$')
ax2.set_ylabel(r'$\beta,\;\beta_{\parallel},\;\xi,\;\hat{B}_{\perp}^{2}$')
ax2.set_ylim(1e-5, 1e2)
ax2.set_yticks([1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2])
ax2.grid(True, which='both', ls='--', alpha=0.3)
ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

# -------------------------
# 図全体レイアウト
# -------------------------
plt.subplots_adjust(right=0.8)  # 右 20 % を凡例スペースに
plt.show()

# ===============================
# 4. 各 r_ratio に対するデータセット (90×3) の作成と出力
# ===============================
# 出力するデータセットの各行は [max_imaginary_CGL, max_imaginary_isotropic, firehose_condition] となる
# ※ CGL の値は、firehose 条件が正の範囲 (r < r_ratios_cutoff) では計算済み。条件が成立しない r_ratio には NaN を補完

# 90 行のデータセットを作成
dataset = np.empty((len(r_ratios), 3))
dataset[:, 1] = max_imaginary_parts_iso          # 等方モデルの値は全 r_ratio に対してある
dataset[:, 2] = firehose_condition               # Firehose 条件

# CGL の値は、cutoff_index まで計算されているので、それ以降は NaN とする
dataset[:cutoff_index, 0] = max_imaginary_parts_CGL
dataset[cutoff_index:, 0] = np.nan

# dataset の各行に対応する r_ratio の値も表示したい場合は、別途 r_ratios を利用するか、カラムとして結合する
# 例: [r_ratio, max_imaginary_CGL, max_imaginary_isotropic, firehose_condition] の 90×4 のデータセット
# 今回は 90×3 (各モデルの値のみ) を出力する

# データセットの出力（例：標準出力に表示）
print("行: r_ratio のインデックス (1～90)")
print("列: [γₘₐₓ/ω₀ (CGL), γₘₐₓ/ω₀ (isotropic), Firehose 条件]")
print(dataset)

# -------------------------
# 追加: 下段プロット用データも含めて保存
# -------------------------
# 8 列: r, γCGL, γiso, Firehose, β∥, β, ξ, B⊥²
full_dataset = np.column_stack([
    r_ratios,
    dataset,                   # 既存の 3 列
    beta_parallel_vals,
    beta_iso_vals,
    xi_vals,
    B_perp_vals
])

# ヘッダー行を列名に合わせて作成
header = ("r_ratio, gamma_CGL, gamma_iso, firehose, "
          "beta_parallel, beta_iso, xi, B_perp_squared")

# 標準出力で確認
print("\n--- full_dataset preview ---")
print(full_dataset[:5])   # 先頭 5 行だけ表示

# CSV 保存
np.savetxt("Fig3a_dataset_full.csv", full_dataset,
           delimiter=",", header=header, comments="")
print(">> Fig3a_dataset_full.csv を保存しました")

