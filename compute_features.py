from operations import compute_gradient, compute_roughness, compute_density, compute_max_local_height_difference, \
    init_pool, compute_verticality, compute_geometric, compute_mean_curvature, compute_gaussian_curvature
from util import pointcloud as pc
from os import path
import argparse
import tkinter as tk
from tkinter import filedialog as fd
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count, shared_memory
import numpy as np
from time import time
from scipy.spatial import cKDTree


def main():
    tk.Tk().withdraw()
    filepath = fd.askopenfilename()
    if filepath == '':
        print("No file selected.")
        return
    elif not path.isfile(filepath) or filepath[-4:] != ".las":
        print("Invalid file selected.")
        return
    filepath = path.abspath(filepath)

    try:
        fchoice = int(input("Do you want to see regular[0] or eigen[1] features?\n"))
    except ValueError:
        print("Invalid selection.")
        return
    if fchoice == 0:
        options = {0, 1, 2, 3, 4, 5, 6}
        menu_blurb = ["Options:", "[0] z_gradient", "[1] roughness", "[2] density", "[3] z_diff", "[4] verticality",
                      "[5] mean curvature", "[6] Gaussian curvature",
                      "Which features would you like to compute?",
                      "Multiple options can be selected if you separate them with spaces. (e.g. 0 3)", ""]
    elif fchoice == 1:
        options = {0, 1, 2, 3, 4, 5, 6, 7, 8}
        menu_blurb = ["Options:", "[0] Sum", "[1] Omnivariance", "[2] Eigenentropy", "[3] Anisotropy",
                      "[4] Planarity", "[5] Linearity", "[6] Surface Variation", "[7] Sphericity", "[8] Verticality",
                      "Which features would you like to compute?",
                      "Multiple options can be selected if you separate them with spaces. (e.g. 0 3)", ""]
    else:
        print("Invalid selection.")
        return
    choice = input('\n'.join(menu_blurb))
    clist = [int(c) for c in choice.split()]
    for c in clist:
        if c not in options:
            print("Invalid selection.")
            return
    results = {}
    tstart = time()
    # setup shared memory space so all processes can access the point data for processing
    points, header = pc.from_las(filepath)
    pc_as_np = points.xyz.to_numpy()
    pc_as_np[:, 0] -= header.offsets[0]
    pc_as_np[:, 1] -= header.offsets[1]
    pc_as_np[:, 2] -= header.offsets[2]
    shm = shared_memory.SharedMemory(create=True, size=pc_as_np.nbytes)
    sharr = np.ndarray(pc_as_np.shape, dtype=pc_as_np.dtype, buffer=shm.buf)
    sharr[:] = pc_as_np[:]
    sharr.flags.writeable = False
    # retrieve xyz once
    pts = points.xyz
    # build nearest neighbor tree
    print("Building KDtree...")
    tree = cKDTree(pts, balanced_tree=False, compact_nodes=False)
    # create process manager
    with ProcessPoolExecutor(max_workers=min(cpu_count()//2, 6), initializer=init_pool,
                             initargs=(shm.name, sharr.shape, sharr.dtype, clist)) as executor:
        temp_idle = list(executor.map(_idle, list(range(cpu_count()))))
        if fchoice == 0:
            for c in clist:
                if c == 0:
                    results["z_gradient"] = compute_gradient(pts, tree, executor, gfield='z', radius=0.5)
                elif c == 1:
                    results["roughness"] = compute_roughness(pts, tree, executor, radius=0.5)
                elif c == 2:
                    results["density"] = compute_density(pts, tree, executor, radius=0.2, precise=True)
                elif c == 3:
                    results["z_diff"] = compute_max_local_height_difference(pts, tree, executor, radius=0.5)
                elif c == 4:
                    results["verticality"] = compute_verticality(pts, tree, executor, radius=0.3)
                elif c == 5:
                    results["mean_curvature"] = compute_mean_curvature(pts, tree, executor, radius=0.3, k=300)
                elif c == 6:
                    results["Gaussian_curvature"] = compute_gaussian_curvature(pts, tree, executor, radius=0.5, k=300)
        else:
            results = compute_geometric(pts, tree, executor, clist, radius=0.5)
    # cleanup shared memory block
    shm.close()
    shm.unlink()
    points.add_fields(**results)
    points.to_las(f"{filepath[:-4]}-processed.las")
    print(f"Finished! Time taken: {time()-tstart}s")


def guimain(s, c, e, r=0.5, k=50):
    tk.Tk().withdraw()
    filepath = fd.askopenfilename()
    if filepath == '':
        print("No file selected.")
        return
    elif not path.isfile(filepath) or filepath[-4:] != ".las":
        print("Invalid file selected.")
        return
    filepath = path.abspath(filepath)
    results = {}
    tstart = time()
    # setup shared memory space so all processes can access the point data for processing
    points, header = pc.from_las(filepath)
    pc_as_np = points.xyz.to_numpy()
    pc_as_np[:, 0] -= header.offsets[0]
    pc_as_np[:, 1] -= header.offsets[1]
    pc_as_np[:, 2] -= header.offsets[2]
    shm = shared_memory.SharedMemory(create=True, size=pc_as_np.nbytes)
    sharr = np.ndarray(pc_as_np.shape, dtype=pc_as_np.dtype, buffer=shm.buf)
    sharr[:] = pc_as_np[:]
    sharr.flags.writeable = False
    # retrieve xyz once
    pts = points.xyz
    # build nearest neighbor tree
    print("Building KDtree...")
    tree = cKDTree(pts, balanced_tree=False, compact_nodes=False)
    # create process manager
    with ProcessPoolExecutor(max_workers=min(cpu_count() // 2, 6), initializer=init_pool,
                             initargs=(shm.name, sharr.shape, sharr.dtype, e)) as executor:
        temp_idle = list(executor.map(_idle, list(range(cpu_count()))))
        for i in s:
            if i == 0:
                results["z_gradient"] = compute_gradient(pts, tree, executor, gfield='z', radius=r)
            elif i == 1:
                results["roughness"] = compute_roughness(pts, tree, executor, radius=r)
            elif i == 2:
                results["density"] = compute_density(pts, tree, executor, radius=r, precise=True)
            elif i == 3:
                results["z_diff"] = compute_max_local_height_difference(pts, tree, executor, radius=r)
            elif i == 4:
                results["verticality"] = compute_verticality(pts, tree, executor, radius=r)
        for i in c:
            if i == 0:
                results["mean_curvature"] = compute_mean_curvature(pts, tree, executor, radius=r, k=k)
            elif i == 1:
                results["Gaussian_curvature"] = compute_gaussian_curvature(pts, tree, executor, radius=0.5, k=k)
            elif i == 2:
                pass
        if len(e) > 0:
            results = {**results, **compute_geometric(pts, tree, executor, e, radius=r, k=k)}
        # cleanup shared memory block
    shm.close()
    shm.unlink()
    points.add_fields(**results)
    points.to_las(f"{filepath[:-4]}-processed.las")
    print(f"Finished! Time taken: {time() - tstart}s")

def cli_main(opt):
    filepath = opt.file
    if filepath == '':
        print("No file selected.")
        return
    elif not path.isfile(filepath) or filepath[-4:] != ".las":
        print("Invalid file selected.")
        return
    filepath = path.abspath(filepath)
    points = pc.from_las(filepath)
    results = {}
    tstart = time()
    # setup shared memory space so all processes can access the point data for processing
    pc_as_np = points.xyz.to_numpy()
    shm = shared_memory.SharedMemory(create=True, size=pc_as_np.nbytes)
    sharr = np.ndarray(pc_as_np.shape, dtype=pc_as_np.dtype, buffer=shm.buf)
    sharr[:] = pc_as_np[:]
    sharr.flags.writeable = False
    # retrieve xyz once
    pts = points.xyz
    # build nearest neighbor tree
    print("Building KDtree...")
    tree = cKDTree(pts, balanced_tree=False, compact_nodes=False)
    # create process manager
    with ProcessPoolExecutor(max_workers=cpu_count(), initializer=init_pool,
                             initargs=(shm.name, sharr.shape, sharr.dtype)) as executor:
        temp_idle = list(executor.map(_idle, list(range(cpu_count()))))
        if opt.gradient:
            results[f"{opt.gfield}_gradient"] = compute_gradient(pts, tree, executor, gfield=opt.gfield)
        if opt.roughness:
            results["roughness"] = compute_roughness(pts, tree, executor, radius=opt.rradius)
        if opt.density:
            results["density"] = compute_density(pts, tree, executor, radius=opt.dradius, precise=(not opt.unprecise))
        if opt.zdiff:
            results["z_diff"] = compute_max_local_height_difference(pts, tree, executor, radius=opt.diffradius)
    # cleanup shared memory block
    shm.close()
    shm.unlink()
    points.add_fields(**results)
    points.to_las(f"{filepath[:-4]}-processed.las")
    print(f"Finished! Time taken: {round(time() - tstart, 3)}s")


def check_args(opt):
    if not opt.gradient and not opt.roughness and not opt.density and not opt.zdiff:
        print("At least one operation needs to be specified.")
        return False
    if opt.gradient and opt.gfield is None:
        print("gradient requires gfield to be specified.")
        return False
    if opt.roughness and opt.rradius is None:
        print("roughness requires rradius to be specified.")
        return False
    if opt.density and opt.dradius is None:
        print("density requires dradius to be specified.")
        return False
    if opt.zdiff and opt.diffradius is None:
        print("zdiff requires diffradius to be specified.")
        return False
    return True


def _idle(n):
    return n ** 2


def interface():
    window = tk.Tk()
    window.title('Feature Processor')
    window.geometry('400x400')

    prompt = tk.Label(master=window, text="Select Features to Compute")
    prompt.pack()

    frame0 = tk.Frame()
    rlabel = tk.Label(master=frame0, text='Radius', width=20)
    rlabel.pack()
    radius = tk.StringVar(value='0.5')
    rentry = tk.Entry(master=frame0, textvariable=radius, width=6)
    rentry.pack()

    Klabel = tk.Label(master=frame0, text='K', width=20)
    Klabel.pack()
    kn = tk.StringVar(value='50')
    kentry = tk.Entry(master=frame0, textvariable=kn, width=6)
    kentry.pack()

    frame0.pack()

    frame1 = tk.Frame()
    frame11 = tk.Frame(master=frame1)
    lcurve = tk.Label(master=frame11, text='Curvature', width=20)
    lcurve.pack()
    frame11.pack()
    frame12 = tk.Frame(master=frame1)
    lstandard = tk.Label(master=frame12, text='Standard', width=20)
    lstandard.pack()
    frame12.pack()

    standardfeatures = {}
    standardfeatures['z diff'] = tk.IntVar()
    tk.Checkbutton(master=frame12, text='z diff', variable=standardfeatures['z diff']).pack()
    standardfeatures['gradient'] = tk.IntVar()
    tk.Checkbutton(master=frame12, text='zgradient', variable=standardfeatures['gradient']).pack()
    standardfeatures['roughness'] = tk.IntVar()
    tk.Checkbutton(master=frame12, text='Roughness', variable=standardfeatures['roughness']).pack()
    standardfeatures['density'] = tk.IntVar()
    tk.Checkbutton(master=frame12, text='Density', variable=standardfeatures['density']).pack()
    standardfeatures['verticality'] = tk.IntVar()
    tk.Checkbutton(master=frame12, text='Verticality', variable=standardfeatures['verticality']).pack()

    curvaturefeatures = {}
    curvaturefeatures['mean'] = tk.IntVar()
    tk.Checkbutton(master=frame11, text='Mean', variable=curvaturefeatures['mean']).pack()
    curvaturefeatures['gauss'] = tk.IntVar()
    tk.Checkbutton(master=frame11, text='Gaussian', variable=curvaturefeatures['gauss']).pack()
    curvaturefeatures['normal'] = tk.IntVar()
    tk.Checkbutton(master=frame11, text='Normal', variable=curvaturefeatures['normal']).pack()

    efeatures = {}
    frame2 = tk.Frame()
    reigen = tk.Label(master=frame2, text='Eigen Features', width=20)
    reigen.pack(side=tk.TOP)

    efeatures['sum'] = tk.IntVar()
    tk.Checkbutton(master=frame2, text='Sum', variable=efeatures['sum']).pack()
    efeatures['omnivariance'] = tk.IntVar()
    tk.Checkbutton(master=frame2, text='Omnivariance', variable=efeatures['omnivariance']).pack()
    efeatures['eigenentropy'] = tk.IntVar()
    tk.Checkbutton(master=frame2, text='Eigenentropy', variable=efeatures['eigenentropy']).pack()
    efeatures['anisotropy'] = tk.IntVar()
    tk.Checkbutton(master=frame2, text='Anisotropy', variable=efeatures['anisotropy']).pack()
    efeatures['planarity'] = tk.IntVar()
    tk.Checkbutton(master=frame2, text='Planarity', variable=efeatures['planarity']).pack()
    efeatures['linearity'] = tk.IntVar()
    tk.Checkbutton(master=frame2, text='Linearity', variable=efeatures['linearity']).pack()
    efeatures['surfacevar'] = tk.IntVar()
    tk.Checkbutton(master=frame2, text='Surface Variation', variable=efeatures['surfacevar']).pack()
    efeatures['sphericity'] = tk.IntVar()
    tk.Checkbutton(master=frame2, text='Sphericity', variable=efeatures['sphericity']).pack()
    efeatures['verticality'] = tk.IntVar()
    tk.Checkbutton(master=frame2, text='Verticality', variable=efeatures['verticality']).pack()

    def submit():
        smap = {'gradient': 0, 'roughness': 1, 'density': 2, 'z diff': 3, 'verticality': 4}
        cmap = {'mean': 0, 'gauss': 1, 'normal': 2}
        emap = {'sum': 0, 'omnivariance': 1, 'eigenentropy': 2, 'anisotropy': 3, 'planarity': 4, 'linearity': 5, 'surfacevar': 6, 'sphericity': 7, 'verticality': 8}
        slist = [smap[f[0]] for f in standardfeatures.items() if f[1].get()]
        clist = [cmap[f[0]] for f in curvaturefeatures.items() if f[1].get()]
        elist = [emap[f[0]] for f in efeatures.items() if f[1].get()]
        print(slist)
        print(clist)
        print(elist)
        window.destroy()
        try:
            irad = float(radius.get())
            ik = int(kn.get())
        except ValueError:
            print('Radius must be a float and k must be an int.')
            exit()
        guimain(slist, clist, elist, r=irad, k=ik)
        exit()
    button = tk.Button(master=window, text="Process", width=20, height=2, command=submit)

    frame1.pack(side=tk.LEFT)
    frame2.pack(side=tk.RIGHT)
    button.pack(side=tk.BOTTOM)

    def userexit():
        print('Nothing selected.')
        window.destroy()
        exit()
    window.protocol("WM_DELETE_WINDOW", userexit)
    window.mainloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str)
    parser.add_argument("--gradient", action="store_true")
    parser.add_argument("--gfield", type=str)
    parser.add_argument("--roughness", action="store_true")
    parser.add_argument("--rradius", type=float)
    parser.add_argument("--density", action="store_true")
    parser.add_argument("--dradius", type=float)
    parser.add_argument("--unprecise", action="store_true")
    parser.add_argument("--zdiff", action="store_true")
    parser.add_argument("--diffradius", type=float)

    opt = parser.parse_args()
    if opt.gradient or opt.roughness or opt.density or opt.zdiff or opt.file is not None:
        if not check_args(opt):
            exit()
        cli_main(opt)
    else:
        interface()
