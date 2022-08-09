import numpy as np
import scipy as sp
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar import vecm


def sample_one(P, sig, T, burn=100):
    q = sig.shape[0]
    y = np.zeros((T + burn + 2, q))
    err = np.random.multivariate_normal(np.zeros(q), sig, T + burn + 2)
    y[0] = 0 + err[0, :]
    y[1] = np.matmul(P, y[0]) + err[1, :]
    for i in range(2, T + burn + 2):
        y[i] = y[i - 1] + np.matmul(P, y[i - 1]) + err[i]
    return y[burn:, :]


def comm_mat(m, n):
    # determine permutation applied by K
    A = np.reshape(np.arange(m * n), (m, n), order="F")
    w = np.reshape(A.T, m * n, order="F")

    # apply this permutation to the rows (i.e. to each column) of identity matrix
    M = np.eye(m * n)
    M = M[w, :]
    return M


def get_lseq(a, k):
    lmax = (0.01 + 0.8) * k / 2
    b = 2 * lmax / k - a
    return [np.sqrt(a + (b - a) * i / (k - 1)) for i in range(k)]


def create_params(k, a=0.01):
    d = 12
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


def est_p(y, r, t):
    y_del = (y - np.roll(y, 1, axis=0))[1:, :]
    y = y[:-1, :]
    s_yy = np.einsum("ni,nj->ij", y, y) / t
    s_dy = np.einsum("ni,nj->ij", y_del, y) / t
    s_yd = s_dy.T
    s_dd = np.einsum("ni,nj->ij", y_del, y_del) / t
    lhs = s_yd.dot(np.linalg.inv(s_dd)).dot(s_dy)
    rhs = s_yy
    lam, g = sp.linalg.eigh(lhs, rhs)
    idx = lam.argsort()[::-1]
    lam, g = lam[idx], g[:, idx]
    return s_dy.dot(g[:, :r]).dot(g[:, :r].T)


def cov(x):
    return np.einsum("ij,ik->jk", x, x) / len(x)


def main():
    sns.set_theme()

    k = 8
    a_seq = [0.001, 0.002, 0.003, 0.004, 0.005]

    # Fixed T
    T = 500
    nsim = 500
    res = []
    for a in a_seq:
        tmp = pd.DataFrame(np.zeros((nsim, 3)))
        tmp.columns = ["rank", "bias", "a"]
        Pi, s, b = create_params(k, a)
        for i in range(nsim):
            y = sample_one(Pi, s, T)
            r = vecm.select_coint_rank(y, -1, 0, method="trace").rank
            tmp.iloc[i, :] = [r, b[max(k - r, 0)], a]
        res.append(tmp)

    res = pd.concat(res, axis=0)
    res["a"] = pd.Categorical(res["a"])
    res.head()

    df_rank = res.groupby(["rank", "a"]).size().reset_index()
    df_rank.rename(columns={0: "freq"}, inplace=True)
    df_rank["freq"] /= nsim

    rr_min, rr_max = 2, 12
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

    plt.figure(figsize=(10, 5))
    sns.set_style("whitegrid")

    palette = ["#008fd5", "#fc4f30", "#e5ae38", "#6d904f", "#8b8b8b"]  # , "#810f7c"]
    plt.figure(figsize=(10, 5))
    fig = sns.lineplot(
        x="rank",
        y="freq",
        hue="a",
        linewidth=1,
        data=df_rank,
        palette=sns.color_palette(palette, 5),
    )
    sns.scatterplot(
        x="rank",
        y="freq",
        hue="a",
        legend=False,
        data=df_rank,
        palette=sns.color_palette(palette, 5),
    )
    plt.axvline(8, 0, 1, alpha=1, color="black", ls="--", linewidth=".7")
    fig.set_ylabel("Proportion")
    fig.set_xlabel("Estimated Rank")
    plt.legend(title="$\lambda_{min}$")
    print("DONE")

    """
    p = sns.lineplot(x="rank", y="bias", hue="del_lam", linewidth=2, data=df_bias)
    p.set_ylabel("b")
    p.set_xlabel("Rank")
    p.get_legend().remove()
    plt.savefig("../plots/fixed_T_bias")
    plt.clf()
    print("DONE")
    """

    # Varying T
    res = []
    a_seq = [0.001, 0.002, 0.003, 0.004, 0.005]
    ts = [100 * i for i in range(3, 22)]
    tmax = np.max(ts)
    nsim = 500
    for a in a_seq:
        Pi, s, b = create_params(k, a)
        for i in range(nsim):
            y = sample_one(Pi, s, tmax)
            for t in ts:
                y_ = y[:t, :]
                r = vecm.select_coint_rank(y_, -1, 0, method="trace").rank
                Pi_hat = est_p(y_, r)
                b_hat = np.linalg.norm((Pi - Pi_hat).T.ravel()) ** 2
                res.append(
                    {
                        "Asymptotic bias": b[max(k - r, 0)],
                        "Empirical bias": b_hat,
                        "Estimator": Pi_hat.T.ravel(),
                        "Rank": r,
                        "a": str(a),
                        "T": t,
                    }
                )

    res = pd.DataFrame(res)
    res.head()

    df_plot = res.groupby(["a", "T"])["Estimator"].apply(
        lambda x: np.trace(cov(np.stack(x) - np.mean(x)))
    )
    df_plot = df_plot.reset_index()
    df_plot.rename(columns={"Estimator": "variance"}, inplace=True)
    df_plot["bias"] = res.groupby(["a", "T"])["Asymptotic bias"].mean().values
    df_plot["MSE"] = res.groupby(["a", "T"])["Empirical bias"].mean().values
    df_plot["rank"] = res.groupby(["a", "T"])["Rank"].mean().values
    df_plot.head()

    plt.figure(figsize=(10, 5))
    p = sns.lineplot(
        x="T",
        y="bias",
        hue="a",
        linewidth=1,
        data=df_plot,
        palette=sns.color_palette(palette, 5),
    )
    p.set_ylabel("Average bias")
    p.set_xlabel("T")
    p.legend(title="$\lambda_{min}$")
    plt.savefig("../plots/range_T_rank")
    plt.clf()
    print("DONE")


if __name__ == "__main__":
    main()
