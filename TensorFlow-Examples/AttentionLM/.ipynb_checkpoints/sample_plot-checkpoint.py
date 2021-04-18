import matplotlib
import matplotlib.pyplot as plt
import numpy as np; np.random.seed(1)

x = np.sort(np.random.random([3, 15]))
y = np.random.random([3, 15])
names = np.array(list("ABCDEFGHIJKLMNO"))

norm = matplotlib.colors.Normalize(vmin=0, vmax=3)
# cmap = plt.cm.rainbow
cmap = plt.cm.RdYlGn
c = np.random.rand(3)
lines = []

fig, ax = plt.subplots()
for i in range(3):
    line, = plt.plot(x[i], y[i], c=cmap(norm(c[i] * 3)), label='line%d'%i)
    lines.append(line)

annot = ax.annotate("", xy=(0,0), xytext=(-20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

def update_annot(line, ind):
    x,y = line.get_data()
    annot.xy = (x[ind["ind"][0]], y[ind["ind"][0]])
    # print(line.get_label())
    text = f"{line.get_label()}"
    annot.set_text(text)
    annot.get_bbox_patch().set_alpha(0.4)


def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        line = None
        for i in range(3):
            cont, ind = lines[i].contains(event)
            if cont:
                line = lines[i]
                break
        if cont:
            update_annot(line, ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", hover)

plt.show()
input('')