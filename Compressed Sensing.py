import numpy as np
from PIL import Image
from scipy.fftpack import dct, idct
import cvxpy as cp
import matplotlib.pyplot as plt


# ===== DCT 2D / IDCT 2D =====
def dct2(block):
    # 2次元DCT（行と列に1次元DCTを適用）
    return dct(dct(block.T, norm='ortho').T, norm='ortho')


def idct2(block):
    # 2次元IDCT（行と列に1次元IDCTを適用）
    return idct(idct(block.T, norm='ortho').T, norm='ortho')


def main():
    # ===== 画像を読み込んで前処理 =====
    # 自分の画像に差し替える場合はここを変える
    img_path = "sample.png"  # ここに自分の画像パス
    N = 64  # 画像サイズ（N x N にリサイズして実験）

    img = Image.open(img_path).convert("L")  # グレースケール
    img = img.resize((N, N), Image.BICUBIC)
    img = np.array(img).astype(np.float32) / 255.0  # 0〜1に正規化

    # ===== DCT空間でスパースとみなす =====
    X_dct = dct2(img)             # DCT係数（N x N）
    x_vec = X_dct.flatten()       # ベクトル化 (N^2,)

    n = x_vec.size  # 次元数 N^2

    # ===== 測定（ランダムサンプリング） =====
    sampling_ratio = 0.3  # サンプリング率 30%（欠損率70%）
    m = int(sampling_ratio * n)  # 測定数

    rng = np.random.default_rng(seed=0)
    idx = rng.choice(n, size=m, replace=False)  # 取る成分のインデックス

    y = x_vec[idx]  # 観測データ y = P x

    # ===== 圧縮センシング復元（ℓ1最小化） =====
    # min ||z||_1  s.t.  z[idx] = y
    z = cp.Variable(n)
    constraints = [z[idx] == y]
    objective = cp.Minimize(cp.norm1(z))
    prob = cp.Problem(objective, constraints)

    print("Solve L1 minimization...")
    prob.solve(verbose=True, solver=cp.SCS)  # solverは環境に合わせて変更可

    x_rec = z.value
    X_dct_rec = x_rec.reshape(N, N)

    # 逆DCTして画像に戻す
    img_rec = idct2(X_dct_rec)
    img_rec = np.clip(img_rec, 0, 1)

    # ===== 結果の表示 =====
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(img, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title(f"CS Reconstructed\n{sampling_ratio*100:.0f}% samples")
    plt.imshow(img_rec, cmap="gray")
    plt.axis("off")

    # 参考：同じサンプリング率で「線形補間」してみる
    x_naive = np.zeros_like(x_vec)
    x_naive[idx] = y  # わかっている成分だけ埋める
    
    # 線形補間で欠損値を補完
    from scipy.interpolate import griddata
    idx_2d = np.unravel_index(idx, (N, N))
    points = np.column_stack(idx_2d)
    values = y
    
    # グリッド座標
    grid_x, grid_y = np.meshgrid(np.arange(N), np.arange(N))
    grid_points = np.column_stack([grid_y.ravel(), grid_x.ravel()])
    
    # 線形補間
    interpolated = griddata(points, values, grid_points, method='linear', fill_value=np.mean(y))
    x_naive_interp = interpolated.reshape(N, N)
    
    X_dct_naive = x_naive_interp
    img_naive = idct2(X_dct_naive)
    img_naive = np.clip(img_naive, 0, 1)

    plt.subplot(1, 3, 3)
    plt.title("Naive (zero-filled)")
    plt.imshow(img_naive, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
