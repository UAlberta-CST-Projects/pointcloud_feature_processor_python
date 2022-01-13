from operations import compute_gradient, compute_roughness, compute_density, compute_max_local_height_difference, init_pool
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

tk.Tk().withdraw()


def main():
    filepath = fd.askopenfilename()
    if filepath == '':
        print("No file selected.")
        return
    elif not path.isfile(filepath) or filepath[-4:] != ".las":
        print("Invalid file selected.")
        return
    filepath = path.abspath(filepath)
    menu_blurb = ["Options:", "[0] z_gradient", "[1] roughness", "[2] density", "[3] z_diff",
                  "Which features would you like to compute?",
                  "Multiple options can be selected if you separate them with spaces. (e.g. 0 3)", ""]
    choice = input('\n'.join(menu_blurb))
    clist = [int(c) for c in choice.split()]
    for c in clist:
        if c not in {0, 1, 2, 3}:
            print("Invalid selection.")
            return
    results = {}
    tstart = time()
    # setup shared memory space so all processes can access the point data for processing
    points = pc.from_las(filepath)
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
        for c in clist:
            if c == 0:
                results["z_gradient"] = compute_gradient(pts, tree, executor, gfield='z', radius=0.3)
            elif c == 1:
                results["roughness"] = compute_roughness(pts, tree, executor, radius=0.3)
            elif c == 2:
                results["density"] = compute_density(pts, tree, executor, radius=0.2, precise=True)
            elif c == 3:
                results["z_diff"] = compute_max_local_height_difference(pts, tree, executor, radius=0.3)
    # cleanup shared memory block
    shm.close()
    shm.unlink()
    points.add_fields(**results)
    points.to_las(f"{filepath[:-4]}-processed.las")
    print(f"Finished! Time taken: {time()-tstart}s")


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
        main()
