"""Remarques

L'entrainement ne semble pas fonctionner: modifier la valeur de la source ne 
change rien aux resultats obtenus. De plus, il semble y avoir un pb avec le 
point (x,y)=(0,0) car les valeurs en ce point different constamment des valeurs 
des autres points et ce dans toutes les situations.
Modif: En mettant une densite gaussienne a la place d'un dirac pour la source
alors quand S!=0 deplacer la source a un impact sur la solution. De plus GPT
m'a dit que numeriquement qque chose de continu est preferable a un Dirac.

Il y a peut etre trop peut de points d'entrainement: si on voulait qu'il y ait 
un point d'entrainement tous les 1 m et toutes les 0.1 s il faudrait
500 * 500 * 100 = 25 000 000 points d'entrainement !

Est-ce que le probleme est bien pose ? Est-ce qu'une condition initiale suffit ?
Intuitivement je dirais que oui mais c'est a verifier.
Modif: En fait ChatGPT m'a dit que pour des equations d'advection diffusion il
faut quand meme imposer des conditions aux limites afin d'avoir une unique
solution meme si le bord du domaine ne correspond a rien en particulier.
Je vais essayer d'imposer la condition que C vaille 0 au bord. Cela
pourrait bien representer le phenomene physique etudie si on choisit
D et u assez faibles pour que sur la periode de temps etudiee le polluant
n'ait pas le temps de sortir du domaine etudie.
"""

import deepxde as dde
import paddle
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import matplotlib.colors

# Definition des parametres
Lx = 500
Ly = 500
u = np.array([0, 0])  # (ux, uy)
D = 50**2 / 10  # coeff de diffusion
T = 10  # temps final
coords_S = np.array([- Lx / 4, Ly / 4])  # coordonnees de la source
S = 0  # valeur de la source
nb_points = 3000  # Ordre de grandeur pour le nombre de points consideres


# Definition des domaines de temps et d'espace
geom = dde.geometry.geometry_2d.Rectangle([-Lx / 2, -Ly / 2], [Lx / 2, Ly / 2])
timedomain = dde.geometry.TimeDomain(0, T)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)


def normal_pdf(z, mu):
    """densite gaussienne reduite (Cov = Id) non normalisee
    z est un array 2d ou chaque ligne correspond a une position spatiale (x,y)"""
    res = np.exp(- np.linalg.norm(z - mu, axis=1)**2 / 2)
    res = paddle.reshape(paddle.to_tensor(res, dtype='float32'), (-1, 1))
    return res


def pde(p, C):
    """residu de l'equadif
    p contient les coordonnes spatiales x, y et le temps t
    ainsi si on s'interesse au point i de coordonnees (xi, yi, ti)
    il est stocke dans p a la ieme ligne et on a donc
    p[i, 0] = xi; p[i, 1] = yi; p[i, 2] = ti
    C correspond a C(p) la concentration en p"""

    dC_x = dde.grad.jacobian(C, p, j=0)
    dC_y = dde.grad.jacobian(C, p, j=1)
    dC_t = dde.grad.jacobian(C, p, j=2)

    dC_xx = dde.grad.hessian(C, p, i=0, j=0)
    dC_yy = dde.grad.hessian(C, p, i=1, j=1)

    return (
        dC_t
        + u[0] * dC_x
        + u[1] * dC_y
        - D * (dC_xx + dC_yy)
        - normal_pdf(p[:, 0:2], coords_S) * S
    )


# Definition des conditions aux limites
nb_source_points = nb_points // 10
times = np.linspace(0, T, nb_source_points)
source_points = np.tile(coords_S, (nb_source_points, 1))
source_points_time = np.c_[
    source_points, times
]  # Points dans le domaine espace temps pour representer la source


def initial_condition(p):
    return normal_pdf(p[:, 0:2], coords_S) * S


ic = dde.icbc.IC(geomtime, initial_condition, lambda _, on_initial: on_initial)


# On concatene tout dans data
data = dde.data.TimePDE(
    geomtime,
    pde,
    ic,
    num_domain=nb_points,
    num_boundary=nb_points // 20,
    num_initial=nb_points // 10,
    anchors=source_points_time,
)


# Definition de l'architecture du reseau de neurones
net = dde.nn.FNN([3] + [20] * 3 + [1], "tanh", "Glorot normal")
model = dde.Model(data, net)


# Entrainement de reseau de neurones
model.compile("adam", lr=1e-3)
losshistory, train_state = model.train(iterations=5000)
# model.compile("L-BFGS")
# losshistory, train_state = model.train()
# dde.saveplot(losshistory, train_state, issave=True, isplot=True)


# Test: on affiche le resultat au temps t
t = 10
x = geom.uniform_points(250 * 250, True)
p = np.c_[x, np.array([t] * x.shape[0])]
y = model.predict(p, operator=pde)

cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "", ["blue", "violet", "red"]
)

resolution_mesh = 300
xi = np.linspace(-Lx / 2, Lx / 2, resolution_mesh)
yi = np.linspace(-Ly / 2, Ly / 2, resolution_mesh)
xi, yi = np.meshgrid(xi, yi)

zi = griddata((x[:, 0], x[:, 1]), y[:, 0], (xi, yi), method="linear")

plt.figure(figsize=(8, 6))
plt.pcolormesh(xi, yi, zi, cmap=cmap, shading="auto")
plt.colorbar(label="Prédiction")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Distribution des prédictions à t={}".format(t))
plt.show()
