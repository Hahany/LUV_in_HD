# calculate radial distribution function
import numpy as np


def PBC(v, lx, ly):  # periodic boundary
    rsLx = lx/2
    rsLy = ly/2

    if v[0] > rsLx:
        v[0] = v[0]-lx
    elif v[0] < -rsLx:
        v[0] = v[0] + lx

    if v[1] > rsLy:
        v[1] = v[1] - ly
    elif v[1] < -rsLy:
        v[1] = v[1] + ly

    return v

def RDFpar(a, t, edges):  # a is structure variable, t is select time frame
    Dt, N, d = np.shape(a.frames)
    ptype = list(set(a.frames[0, :, 1])) #get particle types
    ntype = len(ptype)
    nbond = int(ntype * (ntype + 1) / 2)
    output = np.zeros((len(edges) - 1, nbond+1))
    for i in t:
        flag = 0
        rdf_out = np.zeros((len(edges) - 1, nbond+1))
        total_hist = np.zeros(len(edges)-1)
        # p = np.squeeze(a.frames[i, :, :], 0)
        p = a.frames[i, :, :]
        for ipa in range(N):
            pB = np.delete(p, ipa, axis=0)
            dr_vec = p[ipa, 2:4] - pB[:, 2:4]
            rsDr = dr_vec
            dr_norm = np.linalg.norm(rsDr, axis=1)
            hist, _ = np.histogram(dr_norm, edges)
            total_hist += hist
        total_rdf = total_hist / N
        areaRing = (edges[1:] ** 2 - edges[:-1] ** 2) * np.pi
        f_d = N / (a.L[0]*a.L[1])
        total_rdf = total_rdf / (areaRing * f_d)
        label = []
        for itype in range(ntype):
            for jtype in range(itype, ntype):
                label.append(f'{itype}-{jtype}')
                print(f'pair type: {itype}-{jtype}')
                if itype == jtype:  # P_AA   P_BB  ...
                    # p1 = np.squeeze(a.frames[t, :, :], 0)
                    p1 = a.frames[i, :, :]
                    p1 = p1[p1[:, 1] == ptype[itype]]
                    n1 = p1.shape[0]
                    rdf = np.zeros(len(edges)-1)
                    for ipa in range(n1):
                        pB = np.delete(p1, ipa, axis=0)
                        dr_vec = p1[ipa, 2:4] - pB[:, 2:4]
                        rsDr = dr_vec
                        dr_norm = np.linalg.norm(rsDr, axis=1)
                        hist, _ = np.histogram(dr_norm, edges)
                        rdf += hist
                    rdf = rdf/n1
                    areaRing = (edges[1:]**2 - edges[:-1]**2)*np.pi
                    f_d = N / (a.L[0]*a.L[1])
                    rdf = rdf/(areaRing*f_d)
                    rdf_out[:, flag] = rdf
                else:
                    # p0 = np.squeeze(a.frames[t, :, :], 0)
                    p0 = a.frames[i, :, :]
                    p1 = p0[p0[:, 1] == ptype[itype]]
                    p2 = p0[p0[:, 1] == ptype[jtype]]
                    n1 = p1.shape[0]
                    n2 = p2.shape[0]
                    rdf = np.zeros(len(edges)-1)
                    for ipa in range(n1):
                        dr_vec = p1[ipa, 2:4] - p2[:, 2:4]
                        rsDr = dr_vec
                        dr_norm = np.linalg.norm(rsDr, axis=1)
                        hist, _ = np.histogram(dr_norm, edges)
                        rdf += hist
                    rdf = rdf / n1
                    areaRing = (edges[1:] ** 2 - edges[:-1] ** 2) * np.pi
                    f_d = N / (a.L[0]*a.L[1])
                    rdf = rdf / (areaRing * f_d)
                    rdf_out[:, flag] = rdf
                flag += 1
        rdf_out[:, -1] = total_rdf
        output += rdf_out
    output = output/len(t)
    label.append('Total')
    return output, label











