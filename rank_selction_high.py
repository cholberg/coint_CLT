import numpy as np
import scipy as sp
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
from tqdm import tqdm
from statsmodels.tsa.vector_ar import vecm

sns.set_style("whitegrid")
d = 40
k = 20
n = d - k
T = 200


def sample_one(P, sig, T, burn=100):
    q = sig.shape[0]
    y = np.zeros((T + burn + 2, q))
    err = np.random.multivariate_normal(np.zeros(q), sig, T + burn + 1)
    y[0] = 0
    y[1] = np.matmul(P, y[0]) + err[0]
    for i in range(2, T + burn + 2):
        y[i] = y[i - 1] + np.matmul(P, y[i - 1]) + err[i - 1]
    return y[burn:, :]


def comm_mat(m, n):
    # determine permutation applied by K
    A = np.reshape(np.arange(m * n), (m, n), order="F")
    w = np.reshape(A.T, m * n, order="F")

    # apply this permutation to the rows (i.e. to each column) of identity matrix
    M = np.eye(m * n)
    M = M[w, :]
    return M


def get_psi(i, g):
    if i == 0:
        return np.vstack([np.eye(d), np.zeros((k, d))])
    psi_i = np.linalg.matrix_power(np.eye(k) + g, i - 1)
    psi_i = np.hstack([psi_i, np.zeros((k, n))])
    return np.vstack([g, np.zeros((n, k)), np.eye(k)]).dot(psi_i)


def get_gam(i, g, s, s_til):
    if i >= 1:
        tmp = []
        for j in range(50):
            tmp.append(get_psi(i + j, g).dot(s).dot(get_psi(j, g).T))
        return np.sum(tmp, axis=0)
    elif i <= -1:
        tmp = []
        for j in range(50):
            tmp.append(get_psi(j, g).dot(s).dot(get_psi(j - i, g).T))
        return np.sum(tmp, axis=0)
    else:
        return s_til.copy()


def get_lseq(a, k):
    lmax = (0.01 + 0.8) * k / 2
    b = 2 * lmax / k - a
    return [np.sqrt(a + (b - a) * i / (k - 1)) for i in range(k)]


def create_params(k, a=0.01):
    n = d - k
    # Check that a > 0
    assert a > 0, "a > 0 expected, got a={}".format(a)
    # Create parameters
    beta = np.vstack([np.eye(k), np.zeros((d - k, k))])
    alpha = np.vstack([-2 * np.diag(get_lseq(a, k)), np.zeros((d - k, k))])
    alpha_perp = sp.linalg.orth(
        np.eye(d) - alpha.dot(np.linalg.inv(alpha.T.dot(alpha))).dot(alpha.T)
    )
    s = np.eye(d)
    # s[:k, :k] *= 2
    # s = np.diag(np.arange(d, dtype=np.float64) + 1)

    Q = np.concatenate([beta.T, alpha_perp.T], axis=0)
    Q_inv = np.linalg.inv(Q)
    Pi = alpha.dot(beta.T)
    Gamma = Q.dot(Pi).dot(Q_inv)
    Psi = np.eye(d) + Gamma

    Sigma_x = np.zeros((k, k))
    for i in range(50):
        Psi_i = np.linalg.matrix_power(Psi[:k, :k], i)
        Sigma_x += Psi_i.dot(s[:k, :k]).dot(Psi_i.T)
    Sigma_w_inv = np.linalg.inv(s[k:, k:])

    # Computing Sigma_m
    Gamma_ = Gamma[:k, :k]
    Sigma_del = s.copy()
    Sigma_del[:k, :k] += Gamma_.dot(Sigma_x).dot(Gamma_.T)
    Sigma_del_inv = np.linalg.inv(Sigma_del)[:k, :k]
    A = Sigma_x.dot(Gamma_.T)
    A = A.dot(Sigma_del_inv).dot(A.T)
    Lam, L = sp.linalg.eigh(A, Sigma_x)
    idx = Lam.argsort()[::-1]
    Lam, L = Lam[idx], L[:, idx]

    # Computing bias
    b = [
        np.linalg.norm(
            (
                Gamma[:k, :k].dot(Sigma_x).dot(L[:, (k - i) :]).dot(L[:, (k - i) :].T)
            ).T.ravel()
        )
        ** 2
        for i in range(k + 1)
    ]

    return Pi, s, b


def est_eig(y):
    y_del = (y - np.roll(y, 1, axis=0))[1:, :]
    y = y[:-1, :]
    T = y_del.shape[0]
    s_yy = np.einsum("ni,nj->ij", y, y) / T
    s_dy = np.einsum("ni,nj->ij", y_del, y) / T
    s_yd = s_dy.T
    s_dd = np.einsum("ni,nj->ij", y_del, y_del) / T
    lhs = s_yd.dot(np.linalg.inv(s_dd)).dot(s_dy)
    rhs = s_yy
    lam, g = sp.linalg.eigh(lhs, rhs)
    idx = lam.argsort()[::-1]
    return lam[idx], g[:, idx], s_dy


def boot_sample(res, r, P, B=999):
    T = res.shape[0] + 1
    out = np.zeros(B)
    for b in range(B):
        x_s = np.zeros((T, d))
        idx = np.random.choice(range(T - 1), T - 1)
        res_s = res[idx, :]
        for t in range(1, T):
            x_s[t, :] = P.dot(x_s[t - 1, :]) + res_s[t - 1, :]
        lam_s, _, _ = est_eig(x_s)
        out[b] = -T * np.sum(np.log(1 - lam_s[r:]))
    return out


def rank_test_boot(y, sig=0.05, B=999):
    T = y.shape[0]
    d = y.shape[1]
    lam, g, s_dy = est_eig(y)
    y_del = (y - np.roll(y, 1, axis=0))[1:, :]
    y_ = y[:-1, :]
    for r in range(d):
        q_r = -T * np.sum(np.log(1 - lam[r:]))
        Pi_r = s_dy.dot(g[:, :r]).dot(g[:, :r].T)
        res_r = (y_del.T - Pi_r.dot(y_.T)).T
        res_r += -np.mean(res_r)
        P_r = np.eye(d) + Pi_r
        q_s = boot_sample(res_r, r, P_r, B)
        p_r = np.mean(q_s > q_r)
        # print(p_r)
        if p_r > sig:
            return r
    return d


def boot_sim(a):
    Pi, s, b = create_params(k, a)
    y = sample_one(Pi, s, T)
    r = rank_test_boot(y, B=299)
    return {"rank": r, "bias": b[max(k - r, 0)], "a": a}


def main():
    nsim = 200
    res = []
    a_seq = [0.01, 0.03, 0.1, 0.3]

    with multiprocessing.Pool(processes=8) as pool:
        progress_bar = tqdm(total=len(a_seq) * nsim)
        # res = pool.map(boot_sim, a_seq * nsim)
        print("mapping...")
        results = tqdm(pool.imap(boot_sim, a_seq * nsim), total=len(a_seq) * nsim)
        print("running...")
        res = list(results)
        print("done")

    res = pd.DataFrame(res)
    res["a"] = pd.Categorical(res["a"])

    df_rank = res.groupby(["rank", "a"]).size().reset_index()
    df_rank.rename(columns={0: "freq"}, inplace=True)
    df_rank["freq"] /= nsim

    rr_min, rr_max = 2, 20
    df_bias = []
    for a in a_seq:
        Pi, s, b = create_params(k, a)
        for r in range(rr_min, rr_max + 1):
            if k - r < 0:
                bi = 0
            else:
                bi = b[k - r]
            df_bias.append({"rank": r, "a": a, "bias": bi})
    df_bias = pd.DataFrame(df_bias)
    df_bias["a"] = pd.Categorical(df_bias["a"])

    sns.set_style("whitegrid")

    palette = [
        "#008fd5",
        "#fc4f30",
        "#e5ae38",
        "#6d904f",
    ]  # , "#8b8b8b"]  # , "#810f7c"]
    plt.figure(figsize=(10, 5))
    fig = sns.lineplot(
        x="rank",
        y="freq",
        hue="a",
        linewidth=1,
        data=df_rank,
        palette=sns.color_palette(palette, 4),
    )
    sns.scatterplot(
        x="rank",
        y="freq",
        hue="a",
        legend=False,
        data=df_rank,
        palette=sns.color_palette(palette, 4),
    )
    plt.axvline(k, 0, 1, alpha=1, color="black", ls="--", linewidth=".7")
    fig.set_ylabel("Proportion")
    fig.set_xlabel("Estimated Rank")
    plt.legend(title="$\lambda_{min}$")
    plt.savefig("../plots/fixed_T_rank_high.pdf")
    plt.clf()
    print("DONE")

    p = sns.lineplot(
        x="rank",
        y="bias",
        hue="a",
        linewidth=1,
        data=df_bias,
        palette=sns.color_palette(palette, 4),
    )
    p.set_ylabel("Asymptotic Bias")
    p.set_xlabel("Rank")
    plt.legend(title="$\lambda_{min}$")
    plt.savefig("../plots/fixed_T_bias_high.pdf")
    plt.clf()
    print("DONE")


if __name__ == "__main__":
    main()
