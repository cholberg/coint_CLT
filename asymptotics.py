import warnings
import numpy as np
import scipy as sp
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


k = 2
m = 1  # under
l = 2  # over
d = 4
n = d - k
T = 5000
nsim = 1000


def sample_one(P, sig, T, burn=100):
    q = sig.shape[0]
    y = np.zeros((T + burn + 2, q))
    err = np.random.multivariate_normal(np.zeros(q), sig, T + burn + 2)
    y[0] = 0 + err[0, :]
    y[1] = np.matmul(P, y[0]) + err[1, :]
    for i in range(2, T + burn + 2):
        y[i] = y[i - 1] + np.matmul(P, y[i - 1]) + err[i]
    return y[burn:, :]


def create_params(k, d):
    b = [1, -1] + [0] * (d - 2)
    beta = np.column_stack([np.roll(b, i) for i in range(k)])
    beta_perp = sp.linalg.orth(
        np.eye(d) - beta.dot(np.linalg.inv(beta.T.dot(beta))).dot(beta.T)
    )
    alpha = np.row_stack([np.eye(k)] + [np.zeros(k) for i in range(d - k)]) * -0.7
    alpha_perp = sp.linalg.orth(
        np.eye(d) - alpha.dot(np.linalg.inv(alpha.T.dot(alpha))).dot(alpha.T)
    )
    return beta, beta_perp, alpha, alpha_perp


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


def sim_bm(p, dt=0.0001):
    steps = int(np.ceil(1 / dt))
    u = 1 / steps
    dB = np.sqrt(u) * np.random.normal(size=(steps - 1, p))
    B_init = np.zeros((1, p))
    B = np.concatenate((B_init, np.cumsum(dB, axis=0)), axis=0)
    return B, dB


def sample_asym(Sigma, Sigma_sq, Sigma_inv, Sigma_x, Sigma_x_inv, Xi, C_m):
    # Random walk asymptotics
    w, dw = sim_bm(d)
    w = w[:-1, :]
    u = 1 / w.shape[0]
    int_wdw = np.einsum("ni,nj->ji", w, dw)
    int_wwd = np.einsum("ni,nj->ij", w, w) * u

    D = np.concatenate((np.zeros((k, n)), np.eye(n)), axis=0)
    J = Sigma_sq.dot(int_wdw).dot(Sigma_sq).dot(D)
    J_2 = Sigma[:k, k:].dot(Sigma_inv).dot(J[k:])
    J_1 = J[:k, :] - J_2
    B = D.T.dot(Sigma_sq).dot(int_wwd).dot(Sigma_sq).dot(D)
    B_inv = np.linalg.inv(B)

    lhs = J[k:, :].T.dot(Sigma_inv).dot(J[k:, :])
    rhs = B
    lam_2, l_2 = sp.linalg.eigh(lhs, rhs)
    idx = lam_2.argsort()[::-1]
    lam_2, l_2 = lam_2[idx], l_2[:, idx]
    l_2l = l_2[:, :l].dot(l_2[:, :l].T)

    # Stationary asymptotics
    V = np.random.multivariate_normal(np.zeros(d * k), np.kron(Sigma_x, Sigma))
    V = V.reshape((k, -1)).T
    V_r = V.dot(Sigma_x_inv)
    V_til = np.random.multivariate_normal(np.zeros(d * k), Xi)
    V_til = V_til.reshape(k, -1).T

    # Sample from reduced rank asymptotic distribution
    rr = np.zeros((d, d))
    rr[:, :k] = V_r
    rr[:k, k:] = J_1.dot(B_inv)

    # Sample from underestimated rank asymptotic distribution
    rru = np.zeros((d, d))
    rru[:, :k] = V_til
    rru[:k, k:] = C_m.dot(J_1).dot(B_inv)

    # Sample from overestimated rank asymptotic distribution
    rro = np.zeros((d, d))
    rro[:, :k] = V_r
    rro[:k, k:] = J_1.dot(B_inv) + J_2.dot(l_2l)
    rro[k:, k:] = J[
        k:,
    ].dot(l_2l)

    return rr, rru, rro


def est_diff(est, Pi, T, bi=0):
    est_ = est.copy()
    est_[:k, :k] += -bi  # remove bias
    D = np.diag([np.sqrt(T)] * k + [T] * n)
    return (est_ - Pi).dot(D)


def sample_est(Pi, Sigma, Q, Q_inv, b, T=8000):
    # Defining response and predictor
    y = sample_one(Pi, Sigma, T + 1)
    x = Q.dot(y.T).T
    x_del = (x - np.roll(x, 1, axis=0))[1:, :]
    x = x[:-1, :]
    Gamma = Q.dot(Pi).dot(Q_inv)

    # Covariance matrices
    s_xx = np.einsum("ni,nj->ij", x, x) / T
    s_dx = np.einsum("ni,nj->ij", x_del, x) / T
    s_xd = s_dx.T
    s_dd = np.einsum("ni,nj->ij", x_del, x_del) / T

    # Solving eigenvalue problem
    lhs = s_xd.dot(np.linalg.inv(s_dd)).dot(s_dx)
    rhs = s_xx
    lam, g = sp.linalg.eigh(lhs, rhs)
    idx = lam.argsort()[::-1]
    lam, g = lam[idx], g[:, idx]

    # Computing estimators
    gam_hat_r = s_dx.dot(g[:, :k]).dot(g[:, :k].T)
    gam_hat_m = s_dx.dot(g[:, :m]).dot(g[:, :m].T)
    gam_hat_l = s_dx.dot(g[:, : (k + l)]).dot(g[:, : (k + l)].T)

    gam_hat_r = est_diff(gam_hat_r, Gamma, T)
    gam_hat_m = est_diff(gam_hat_m, Gamma, T, b)
    gam_hat_l = est_diff(gam_hat_l, Gamma, T)

    return gam_hat_r, gam_hat_m, gam_hat_l


def make_plot_df(dat, r, dname):
    n = dat.shape[0]
    df_plt = pd.DataFrame(dat.reshape((n, -1)))
    df_plt["distribution"] = dname
    df_plt["rank"] = r
    return df_plt


def plot_mult_marginals(dat_tuple):
    dns = np.unique([dn for _, dn, _ in dat_tuple])
    rs = np.unique([r for r, _, _ in dat_tuple])

    clrs = ["#008fd5", "#fc4f30", "#e5ae38", "#6d904f", "#8b8b8b", "#810f7c"]
    clrs = {rs[i]: clrs[i] for i in range(len(rs))}
    styles = ["-", "--", "-."]
    styles = {dns[i]: styles[i] for i in range(len(dns))}

    df_plts = []
    for r, n, dat in dat_tuple:
        df_plts.append(make_plot_df(dat, r, n))
    df_plt = pd.melt(pd.concat(df_plts), id_vars=["distribution", "rank"])
    df_plt.rename(columns={"variable": "coord"}, inplace=True)
    df_plt["col"] = df_plt["coord"] % d
    df_plt["row"] = df_plt["coord"] // d

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(d, d, sharey=False, figsize=(15, 15))
    for i in range(d):
        for j in range(d):
            ls = []
            ns = []
            for dn in dns:
                ls.append(plt.Line2D([0], [0], color="black", ls=styles[dn], lw=1))
                ns.append(dn)
                lc = []
                nc = []
                for r in rs:
                    lc.append(plt.Line2D([0], [0], color=clrs[r], lw=1))
                    nc.append(str(r))
                    df_tmp = df_plt.loc[
                        (df_plt["row"] == i)
                        & (df_plt["col"] == j)
                        & (df_plt["distribution"] == dn)
                        & (df_plt["rank"] == r)
                    ]
                    sns.kdeplot(
                        x="value",
                        color=clrs[r],
                        ls=styles[dn],
                        data=df_tmp,
                        ax=ax[i, j],
                    )
            ax[i, j].set_ylabel("")
            ax[i, j].set_xlabel("")
            if min(i, j) >= k:
                ax[i, j].set_ylim(0.0, 0.12)
    legend_1 = fig.legend(lc, nc, loc=(0.55, 0.015), prop={"size": 12})
    legend_1.set_title("Rank", prop={"size": 12})
    legend_2 = fig.legend(ls, ns, loc=(0.41, 0.025), prop={"size": 12})
    legend_2.set_title("Distribution", prop={"size": 12})
    fig.add_artist(legend_1)
    plt.savefig("../plots/dist.pdf", bbox_inches="tight")
    plt.clf()


def main():
    beta, beta_perp, alpha, alpha_perp = create_params(k, d)
    Q = np.concatenate([beta.T, alpha_perp.T], axis=0)
    Q_inv = np.linalg.inv(Q)
    Pi = alpha.dot(beta.T)
    Gamma = Q.dot(Pi).dot(Q_inv)
    Psi = np.eye(d) + Gamma
    Sigma = np.random.uniform(size=(d, d))
    Sigma = np.eye(d) * 0.5 + Sigma.dot(Sigma.T)
    Sigma_w = Q.dot(Sigma).dot(Q.T)
    Sigma_x = np.zeros((k, k))
    for i in range(50):
        Psi_i = np.linalg.matrix_power(Psi[:k, :k], i)
        Sigma_x += Psi_i.dot(Sigma_w[:k, :k]).dot(Psi_i.T)
    Sigma_x_inv = np.linalg.inv(Sigma_x)
    Sigma_w_sq = sp.linalg.sqrtm(Sigma_w)
    Sigma_w_inv = np.linalg.inv(Sigma_w[k:, k:])

    # Computing Sigma_m
    Gamma_ = Gamma[:k, :k]
    Gamma_inv = np.linalg.inv(Gamma_)
    Sigma_del = Sigma_w.copy()
    Sigma_del[:k, :k] += Gamma_.dot(Sigma_x).dot(Gamma_.T)
    Sigma_del_inv = np.linalg.inv(Sigma_del)[:k, :k]
    A = Sigma_x.dot(Gamma_.T)
    A = A.dot(Sigma_del_inv).dot(A.T)
    Lam, L = sp.linalg.eigh(A, Sigma_x)
    idx = Lam.argsort()[::-1]
    Lam, L = Lam[idx], L[:, idx]
    Sigma_m = L[:, :m].dot(L[:, :m].T)
    Sigma_rm = L[:, m:].dot(L[:, m:].T)

    # Computing C_m
    C_m = Gamma_.dot(Sigma_x).dot(Sigma_m).dot(Gamma_inv)

    # Computing bias
    b = Sigma_w[:k, :k] - Sigma_w[:k, k:].dot(Sigma_w_inv).dot(Sigma_w[k:, :k])
    b = b.dot(np.linalg.inv(Gamma[:k, :k].T)) + Gamma[:k, :k].dot(Sigma_x)
    b = -b.dot(L[:, m:]).dot(np.diag(Lam[m:])).dot(L[:, m:].T)

    # Computing Xi
    Sigma_til = np.zeros((d + k, d + k))
    Sigma_til[:d, :d] = Sigma_del.copy()
    Sigma_til[d:, :k] = Sigma_x.dot(Gamma_.T)
    Sigma_til[:k, d:] = Gamma_.dot(Sigma_x)
    Sigma_til[d:, d:] = Sigma_x.copy()
    S_xdel = Sigma_til[d:, :d]
    S_xdid = S_xdel.dot(np.linalg.inv(Sigma_del))
    Xi = np.kron(
        get_gam(0, Gamma_, Sigma_w, Sigma_til), get_gam(0, Gamma_, Sigma_w, Sigma_til)
    )
    for i in range(1, 20):
        Xi += np.kron(
            get_gam(i, Gamma_, Sigma_w, Sigma_til),
            get_gam(i, Gamma_, Sigma_w, Sigma_til),
        ) + np.kron(
            get_gam(-i, Gamma_, Sigma_w, Sigma_til),
            get_gam(-i, Gamma_, Sigma_w, Sigma_til),
        )
    Xi += Xi.dot(comm_mat(d + k, d + k))
    xi = np.zeros((m, d * k, (d + k) ** 2))
    for i in range(m):
        P_i = np.outer(L[:, i], L[:, i])
        F_i = np.hstack(
            [
                np.kron(S_xdid, np.hstack([-S_xdid, np.eye(k)])),
                np.kron(np.eye(k), np.hstack([S_xdid, -Lam[i] * np.eye(k)])),
            ]
        )
        xi_tmp = np.zeros((k**2, (d + k) ** 2))
        for j in range(k):
            if j != i:
                P_j = np.outer(L[:, j], L[:, j])
                l_ij = 1 / (Lam[i] - Lam[j])
                xi_tmp += l_ij * (np.kron(P_i, P_j) + np.kron(P_j, P_i)).dot(
                    F_i
                ) - np.kron(
                    np.hstack([np.zeros((k, d)), P_i]),
                    np.hstack([np.zeros((k, d)), P_i]),
                )
        xi[i] = np.kron(np.eye(k), S_xdel.T).dot(xi_tmp)
        xi[i] += np.kron(
            np.hstack([np.zeros((k, d)), P_i]), np.hstack([np.eye(d), np.zeros((d, k))])
        )
    xi = np.sum(xi, axis=0)
    Xi = xi.dot(Xi).dot(xi.T)

    # Sampling from asypmtotic distributions
    rr_asym = np.empty((nsim, d, d))
    rru_asym = np.empty((nsim, d, d))
    rro_asym = np.empty((nsim, d, d))
    for i in range(nsim):
        s = sample_asym(Sigma_w, Sigma_w_sq, Sigma_w_inv, Sigma_x, Sigma_x_inv, Xi, C_m)
        rr_asym[i], rru_asym[i], rro_asym[i] = s[0], s[1], s[2]

    # Sampling estimators
    rr_est = np.empty((nsim, d, d))
    rru_est = np.empty((nsim, d, d))
    rro_est = np.empty((nsim, d, d))
    for i in range(nsim):
        rr_est[i], rru_est[i], rro_est[i] = sample_est(Pi, Sigma, Q, Q_inv, b, T)

    # Plotting distributions
    dat_list = [
        (2, "finite sample", rr_est),
        (2, "asymptotic", rr_asym),
        (4, "finite sample", rro_est),
        (4, "asymptotic", rro_asym),
        (1, "finite sample", rru_est),
        (1, "asymptotic", rru_asym),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plot_mult_marginals(dat_list)


if __name__ == "__main__":
    main()
