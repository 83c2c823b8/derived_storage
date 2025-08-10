#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def sph_to_cart(R, theta_deg, phi_deg):
    th = np.deg2rad(theta_deg)
    ph = np.deg2rad(phi_deg)
    return np.array([R*np.sin(th)*np.cos(ph),
                     R*np.sin(th)*np.sin(ph),
                     R*np.cos(th)], dtype=float)

def cart_to_angles(vec):
    x, y, z = vec
    r = np.linalg.norm(vec)
    if r == 0:
        return 0.0, 0.0
    theta = np.degrees(np.arccos(z / r))
    phi = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0
    return theta, phi

def normalize(v):
    n = np.linalg.norm(v)
    return v if n == 0 else v / n

def slerp_arc(p, q, n=160):
    u, v = normalize(p), normalize(q)
    dot = np.clip(u @ v, -1.0, 1.0)
    w = np.arccos(dot)
    if np.isclose(w, 0):
        return np.repeat(u[None, :], n, axis=0)
    t = np.linspace(0.0, 1.0, n)
    sw = np.sin(w)
    a = np.sin((1 - t) * w)[:, None] / sw
    b = np.sin(t * w)[:, None] / sw
    return a * u + b * v

def gc(ax, theta1, phi1, theta2, phi2, R=3.0, n=160, **plot_kwargs):
    p = sph_to_cart(R, theta1, phi1)
    q = sph_to_cart(R, theta2, phi2)
    arc = slerp_arc(p, q, n=n) * R
    ax.plot(arc[:,0], arc[:,1], arc[:,2], **plot_kwargs)

def centroid_on_sphere(angles, R=3.0):
    vecs = [sph_to_cart(1.0, th, ph) for th, ph in angles]
    m = np.mean(vecs, axis=0)
    return normalize(m) * R

def plot_sphere_translucent(ax, R=3.0, nu=80, nv=40, color="lightblue", alpha=0.30):
    u = np.linspace(0, 2*np.pi, nu)
    v = np.linspace(0, np.pi, nv)
    xs = R * np.outer(np.cos(u), np.sin(v))
    ys = R * np.outer(np.sin(u), np.sin(v))
    zs = R * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs,
                    rstride=1, cstride=1,
                    linewidth=0, edgecolor="none",
                    shade=True, color=color, alpha=alpha)

def main():
    R = 3.0
    theta_base = -109.5
    N = (0.0, 0.0)
    A = (theta_base, 0.0)
    B = (theta_base, 120.0)
    C = (theta_base, 240.0)

    G_NAB = centroid_on_sphere([N, A, B], R)
    G_NBC = centroid_on_sphere([N, B, C], R)
    G_NCA = centroid_on_sphere([N, C, A], R)
    G_ABC = centroid_on_sphere([A, B, C], R)

    Gth_NAB, Gph_NAB = cart_to_angles(G_NAB)
    Gth_NBC, Gph_NBC = cart_to_angles(G_NBC)
    Gth_NCA, Gph_NCA = cart_to_angles(G_NCA)
    Gth_ABC, Gph_ABC = cart_to_angles(G_ABC)

    fig = plt.figure(figsize=(6.6, 6.6), dpi=140)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=30, azim=130)

    plot_sphere_translucent(ax, R=R, color="lightblue", alpha=0.30)

    for th, ph in [N, A, B, C]:
        x, y, z = sph_to_cart(R, th, ph)
        ax.scatter(x, y, z, s=32, depthshade=False, color="black")

    for v in [G_NAB, G_NBC, G_NCA, G_ABC]:
        ax.scatter(v[0], v[1], v[2], s=32, depthshade=False, color="blue")

    edge_kwargs = dict(color="black", lw=1.8)
    gc(ax, *N, *A, R, **edge_kwargs)
    gc(ax, *N, *B, R, **edge_kwargs)
    gc(ax, *N, *C, R, **edge_kwargs)
    gc(ax, *A, *B, R, **edge_kwargs)
    gc(ax, *B, *C, R, **edge_kwargs)
    gc(ax, *C, *A, R, **edge_kwargs)

    red_kwargs = dict(color="lightblue", lw=2.2)
    gc(ax, *N, Gth_NAB, Gph_NAB, R, **red_kwargs)
    gc(ax, *A, Gth_NAB, Gph_NAB, R, **red_kwargs)
    gc(ax, *B, Gth_NAB, Gph_NAB, R, **red_kwargs)

    gc(ax, *N, Gth_NBC, Gph_NBC, R, **red_kwargs)
    gc(ax, *B, Gth_NBC, Gph_NBC, R, **red_kwargs)
    gc(ax, *C, Gth_NBC, Gph_NBC, R, **red_kwargs)

    gc(ax, *N, Gth_NCA, Gph_NCA, R, **red_kwargs)
    gc(ax, *C, Gth_NCA, Gph_NCA, R, **red_kwargs)
    gc(ax, *A, Gth_NCA, Gph_NCA, R, **red_kwargs)

    gc(ax, *A, Gth_ABC, Gph_ABC, R, **red_kwargs)
    gc(ax, *B, Gth_ABC, Gph_ABC, R, **red_kwargs)
    gc(ax, *C, Gth_ABC, Gph_ABC, R, **red_kwargs)

    ax.set_box_aspect((1,1,1))
    ax.set_xlim(-R, R); ax.set_ylim(-R, R); ax.set_zlim(-R, R)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    plt.show()

if __name__ == "__main__":
    main()
