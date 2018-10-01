# -*- coding: utf-8 -*-
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull
import math
#import pylab as plt
#from matplotlib.collections import LineCollection


def alpha_shape(points, alpha, hull_simplices=[]):

    def add_edge(edges, edge_points, points, i, j):
        if (i, j) in edges or (j, i) in edges:
            return
        edges.add( (i, j) )
        edge_points.append(points[ [i, j] ])
        
    tri = Delaunay(points)
    edges = set()
    edge_points = []    
    instancias = set()   
    # loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.simplices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        if ia not in hull_simplices and ib not in hull_simplices and ic not in hull_simplices:
            # Lengths of sides of triangle
            a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
            b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
            c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)
        
            # Semiperimeter of triangle
            s = (a + b + c)/2.0
        
            # Area of triangle by Heron's formula
            area = math.sqrt(s*(s-a)*(s-b)*(s-c))
            circum_r = a*b*c/(4.0*area)
        
            # Here's the radius filter.
            if (circum_r < 1.0/alpha):
                add_edge(edges, edge_points, points, ia, ib)
                add_edge(edges, edge_points, points, ib, ic)
                add_edge(edges, edge_points, points, ic, ia)
                instancias.add(ia)
                instancias.add(ib)
                instancias.add(ic)
    
    a = []
    b = []
    for k in instancias:
        a.append(points[k])
        b.append(k)
    
    instancias = np.array(a)
    inds = np.array(b)
    
    return (instancias, inds, edge_points)


def alpha_compaction(points, alpha, threshold):
    iterations=0
    inst = np.copy(points)
    edge_points=[]
    inst, inds, edge_points = alpha_shape(inst, alpha)
    while len(inst)>threshold:
        iterations=iterations+1
        inst, inds, edge_points = alpha_shape(inst, alpha, ConvexHull(inst).simplices)
        
        #print('Iteration[{0}] Alpha={1} Instancias={2}'.format(iterations, str(alpha), str(len(inst))))
        #alpha=alpha+0.1
        
    return inst, inds, edge_points

'''
#generating data
mean = (1, 2)
cov = [[1, 0], [0, 1]]
points = np.random.multivariate_normal(mean, cov, 100)
CP=0.2
threshold = int( len(points)*(1-CP) ) 
alpha = 0.5

inst, edge_points = alpha_compaction(points, alpha, threshold)

lines = LineCollection(edge_points)
plt.figure()
plt.title('Alpha={0} Compaction={1}'.format(alpha, CP))
plt.gca().add_collection(lines)

plt.plot(points[:,0], points[:,1], 'o', hold=1, color='#f16824')
plt.plot(inst[:,0], inst[:,1], 'o', hold=1, color='red')
plt.show()
'''

'''
#testing for different values of alpha
for i in range(2):
    alpha = (i+1)*.5
    edge_points = alpha_shape(points, alpha=alpha)
    threshold = int( len(edge_points)*(1-CP) )     
    
    lines = LineCollection(edge_points)
    pl.figure()
    pl.title('Alpha={0} No Compaction'.format(alpha))
    pl.gca().add_collection(lines)
    pl.plot(points[:,0], points[:,1], 'o', hold=1, color='#f16824')
    
    final_edge_points = [i for j, i in enumerate(edge_points) if j not in simplices[0:threshold]]
    lines = LineCollection(final_edge_points)
    pl.figure()
    pl.title('Alpha={0} With Compaction'.format(alpha))
    pl.gca().add_collection(lines)
    pl.plot(points[:,0], points[:,1], 'o', hold=1, color='#f16824')
    
    #alpha_compaction(edge_points, tri.simplices)         
'''
