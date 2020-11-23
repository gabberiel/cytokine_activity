#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 16:30:00 2020

@author: Wojciech Chacholski

Copyright Wojciech chacholski, 2020
This software is to be used only for activities related  with TDA group at KTH 
and TDA course SF2956 at KTH
"""

import numbers
import numpy as np
inf=float("inf")

import pandas as pd

import scipy.spatial as spatial

import matplotlib.pyplot as plt
plt.style.use('ggplot')

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

from ripser import ripser

##########################################
########           pcf            ########
##########################################

class pcf(object):
    
    
    def __init__(
            self, 
            a):
        self.content = np.asarray(a, dtype=float)

                
    def evaluate(
            self, 
            x):
        V=np.concatenate(([0],self.content[1]))
        x_place=np.searchsorted(self.content[0], x, side = 'right')
        return V[x_place]

    def restrict(
            self, 
            start, 
            end):  
        s_place = np.searchsorted(self.content[0], start,
                                  side = 'right')
        if (s_place > 0 and 
            end < inf):
            e_place = np.searchsorted(self.content[0], end)
            V = self.content[:, s_place:e_place]
            return np.concatenate(([[start], [self.evaluate(start)]],
                                   V, [[end], [0]]), axis = 1)
        elif (s_place == 0 and
              end < inf):
            e_place = np.searchsorted(self.content[0], end)
            V = self.content[:, s_place:e_place]
            return np.concatenate((V, [[end], [0]]), axis = 1)
        elif (s_place > 0 and
              end == inf):
            V = self.content[:, s_place:]
            return np.concatenate(([[start], [self.evaluate(start)]],
                                   V), axis = 1)
        else:
            return self.content


    def extend(
            self,
            D):
        dom = self.content[0]
        domain = np.unique(np.concatenate([dom, D]))
        parameters = np.searchsorted(dom, domain, side = 'right')-1
        val = np.append(self.content[1],0)[parameters]
        return np.vstack((domain, val))

    def simplify(
            self):
        C = self.content[1][:-1]-self.content[1][1:]
        k = np.insert(np.where(C!=0)[0] + 1, 0, 0)
        return self.content[:, k]   
            
    def plot(
            self,
            start = -inf,
            end = inf, 
            color = "no",
            linewidth = 1, 
            ext_l = 0,
            ext_r = 0.1):       
        B = self.restrict(start, end)
        if (ext_l == 0 and 
            ext_r == 0):
            B = B
        elif (ext_l == 0 and 
              ext_r > 0):   
            C = [[B[0, -1] + ext_r], 
            [B[1, -1]]]   
            B=np.concatenate((B, C), axis = 1)
        elif (ext_l > 0 and 
              ext_r == 0):
            A = [[B[0, 0] - ext_l], 
                 [0]]
            B=np.concatenate((A, B), axis = 1)
        else:
            A = [[B[0, 0] - ext_l], 
                 [0]]
            C = [[B[0, -1] + ext_r], 
                 [B[1, -1]]]        
            B=np.concatenate((A, B, C), axis = 1)
        if color == "no":
            return plt.step(B[0],B[1],
                            linewidth = linewidth,
                            where = 'post')
        else:
            return plt.step(B[0], B[1],
                            color = color, linewidth = linewidth,
                            where = 'post')        

    def __add__(
            self,
            other):
        if isinstance(other, numbers.Real):
            C = self.content + np.array([[0], [other]])
            if (isinstance(self, pcnif) and 
                other >= 0):
                return pcnif(C)  
            else:
                return pcf(C)   
        elif isinstance(other, pcf):
            F = self.extend(other.content[0])
            G = other.extend(self.content[0]) 
            C = np.vstack((F[0], F[1] + G[1]))
            if (isinstance(self, pcnif) and
                isinstance(other, pcnif)):  
                return pcnif(C)
            else:
                return  pcf(C)
        else:
            raise ValueError("""we can only add to pcf a 
                             pcf or a real number""")        

    def __radd__(
            self, 
            other):
        return self + other
    
    def __neg__(
            self):
        return pcf(np.vstack((self.content[0], -self.content[1])))    

    def __mul__(
            self, 
            other):
        if isinstance(other, numbers.Real):
            C = self.content * np.array([[1], [other]])
            if other > 0:
                if isinstance(self, pcnif):
                    return pcnif(C)
                else:
                    return pcf(C)
            elif other < 0:
                return pcf(C)
            else:
                if isinstance(self, pcnif):
                    return pcnif([[0],[0]])
                else:
                    return pcf([[0],[0]])
        elif isinstance(other, pcf):
            F = self.extend(other.content[0])
            G = other.extend(self.content[0])
            C = np.vstack((F[0], F[1] *  G[1]))
            if (isinstance(self, pcnif) and  
                  isinstance(other, pcnif)):
                return pcnif(C)
            else:
                return  pcf(C)
        else:
            raise ValueError("""we can only multiply a pcf 
                             by a pcf or a real number""")            
       
    def __rmul__(
            self,
            other):
        return self * other

    def __sub__(
            self, 
            other):
        if isinstance(other, numbers.Real):
            C = self.content - np.array([[0], [other]])
            return pcf(C)
        elif  isinstance(other,pcf):
            F = self.extend(other.content[0])
            G = other.extend(self.content[0])
            C = np.vstack((F[0], F[1] - G[1]))            
            return  pcf(C)
        else:
            raise ValueError("""we can substract from a pcf  
                             a pcf or a number""")

    def __rsub__(
            self,
            other):
        if isinstance(other, numbers.Real):
            C = np.array([[0], [other]]) - self.content  
            return pcf(C)
        else:
            raise ValueError("""we can subtract only real nunmbers 
                             and pcfs from pcfs""")

    def __pow__(
            self,
            p):
        C=np.vstack((self.content[0], self.content[1] ** p))
        if (isinstance(self, pcnif) and 
            p >= 1):
            return  pcnif(C)
        else:
            return pcf(C)

    def __abs__(
            self):
        C=np.vstack((self.content[0], np.absolute(self.content[1])))
        return pcf(C)

    def integrate(
            self, 
            start = -inf, 
            end = inf):  
        if start == end:
            return 0
        elif start < end:
            C = self.restrict(start, end)
        else:
            C = self.restrict(end, start)
        if C[1][-1] > 0:
            return inf
        elif C[1][-1] < 0:
            return -inf
        else:
            return sum(np.diff(C[0]) * C[1][:-1])

    def lp_distance(
            self,
            other,
            p=1,
            start = -inf,
            end = inf):
        if (end == inf and 
            self.content[1][-1] != other.content[1][-1]):
            return inf
        else:
            return (abs(self - other) ** p).integrate(start, end) ** (1 / p)      
        
    def approx(
            self,
            precision = -2,
            base = 10):
        precision = np.int_(precision)
        base = np.int_(base)
        if precision >= 0:
            step = base ** precision
        else:
            step = 1 / base**(-precision)    
        first = np.int_(np.floor(np.divide(self.content[0][0], step)))
        last = np.int_(np.ceil(np.divide(self.content[0][-1], step)))
        D=np.arange(first, last+1) * step
        ind = np.searchsorted(self.content[0], D, side="right")-1
        return np.vstack((D, self.content[1][ind]))
           




##########################################
########          pcnif          #########
##########################################

class pcnif(pcf):               

    
    def __init__(
            self, 
            a):
        super().__init__(a)
 

    def interl(
            self,
            other):
        if self.content[1][-1] < other.content[1][-1]:
            return inf
        else:
            FD = self.content[0]
            FV = self.content[1]
            GD = other.content[0]
            GV = other.content[1]
            intervals = np.array([]) 
            i = 0
            while i < len(GD)-1:
                I = np.searchsorted(-FV, -GV[i], side='right')
                intervals = np.append(intervals, GD[i+1]-FD[I])
                i += 1
            return np.amax(intervals)
            


    def interleaving_distance(
            self,
            other):
        return max(self.interl(other), other.interl(self))
    
##########################################
########           bc           ########## -- BARCODE
##########################################

class bc(object):
        
    def __init__(
            self,
            input):
        self.bars = np.asarray(input)
            
            
    def length(self):
        if len(self.bars) == 0:
            outcome = np.nan
        else:
            g = lambda x:  x[1]-x[0]
            outcome = np.apply_along_axis(g, 1, self.bars)
        return outcome

    def stable_rank(
            self):    
        if len(self.bars)==0:
            return pcnif(np.array([[0], [0]]))     
        else:                    
            length = self.length()
            sortlength = np.unique(length, return_counts = True)
            dom = sortlength[0]
            values=np.cumsum(sortlength[1][::-1])[::-1]
            if dom[-1] == inf:
                dom = np.insert(dom[:-1], 0, 0)
            else:
                dom = np.insert(dom, 0, 0) 
                values = np.append(values, 0)    
            return pcnif(np.vstack((dom, values))) 

    def plot(self):    
        plt.yticks([])
        B = self.bars
        L = len(B)
        if L>0:           
            m = np.amax(B[B!= np.inf])
            ind_fin = np.isfinite(B).all(axis=1)
            bars_fin = B[ind_fin]
            pos_fin = np.arange(0, len(bars_fin))
            plt.hlines(pos_fin,bars_fin[:,0],bars_fin[:,1],
                       colors = ["red"], linewidth = 0.6)
            ind_inf = np.isinf(B).any(axis=1)
            bars_inf = B[ind_inf]
            pos_inf = np.arange(len(bars_fin) , 
                                len(bars_fin ) + len(bars_inf))
            ends = np.ones(len(bars_inf)) * 2 * m
            plt.hlines(pos_inf, bars_inf[:, 0], ends, 
                       colors=["blue"], linewidth = 0.6)
        else:
            plt.text(0.2, 0.2, "empty bar code")
        
##########################################
########        distance           #######
##########################################        

class distance(object): 
    
    def __init__(
            self, 
            d,
            limit = 2):
        den = np.asarray(d)
        if den.size == int(0):
            self.size = 1
            self.content = np.array([0])
            self.type = "compressed"
        else:
            m=np.amax(den[den != inf])
            den[den == inf] = limit * m
            if den.ndim == 1:
                self.size = int((1 + np.sqrt(1 + 8*len(den))) / 2)
                self.content = den[0:int(self.size * (self.size - 1) /  2)]
                self.type = "compressed"
            elif (den.ndim == 2 and 
                  den.shape[0] == den.shape[1] and 
                  den.shape[0] >= 1):            
                self.size = int(den.shape[0])
                self.content = den
                self.type = "square"
            else:
                raise ValueError("""The input should be either 
                                 1d array like or 2d square array like""")

    def square_form(
            self):  
        if self.type == "square":
            return self.content
        else:
            C = np.empty([self.size, self.size], dtype=float)
            i = 0
            s = self.size
            while i < s:
                C[i, i] = 0
                j = i + 1
                while j < s:
                    C[i, j] = self.content[int((s * (s - 1) /2 ) 
                                               - ((s - i) * (s - i - 1) / 2) 
                                               + j - i - 1)]
                    C[j, i]=C[i, j]
                    j += 1
                i += 1
            return C

    def compressed_form(
            self):
        if self.type == "compressed":
            return self.content
        else:
            i = 0
            L = np.array([])
            while i < self.size:
                L = np.concatenate((L, self.content[i, i + 1:self.size]))
                i += 1
            return L

    
            
    def sampling(self, 
                 number_instances, 
                 sample_size):
        # Step 0: forming instances:
        instances = np.arange(number_instances)
        I = pd.Index(instances, name = "instances")  
        out = pd.Series(index = I, dtype = object)
        if sample_size > self.size:
            for i in I:
                out[i] = "no_sample"
        else:
            for i in I:
                out[i] = np.sort(np.random.choice(self.size,   
                                                  sample_size, 
                                                  replace = False))        
        return out

    def dend(self,
             clustering_method):
        L = linkage(self.compressed_form(), clustering_method)
        dn=dendrogram(L, orientation='right')
        return dn
        
 
    def global_h0_sr(self,
                     clustering_method):
        L = linkage(self.compressed_form(), clustering_method)
        D = np.array([0])
        val = np.array([self.size])
        i = 0
        j = 0
        S = self.size
        while i < len(L[:, 2]):
            d = L[i, 2]
            S = S - 1
            if d > D[j]:
                D = np.append(D,d)
                val = np.append(val, S )
                j += 1
            else:
                val[j] =val[j]-1
            i += 1
        return pcnif(np.vstack((D, val)))   
       
             

    def h0_sr(self, 
              sampling = "no", 
              clustering_method = "single"):      
        # averaging case
        if isinstance(sampling, pd.core.series.Series):
            square_form = self.square_form()

            instances = sampling.index
            n_instances = len(instances)
            SR = pd.Series(index = instances, dtype = object)  
            for inst in instances:
                SI = sampling.loc[inst]
                if (isinstance(SI, str) and 
                    SI == "no_sample"):
                    SR.loc[inst] = pcnif([[0],[0]])
                else:
                    ec = distance(square_form[np.ix_(SI, SI)])
                    SR.loc[inst] = ec.global_h0_sr(clustering_method)
                out = np.sum(SR) * (1/ n_instances)
            return out
        
        # Global, no averaging
        if (isinstance(sampling, str) and 
            sampling == "no"):
            out = self.global_h0_sr(clustering_method)
            return out
        raise ValueError("""The sampling has to be either the string "no" or a 
                         result of the samplig method""")
  
    def global_bc(self,
                  maxdim = 1,
                  thresh = inf,
                  coeff = 5):
        dgms=ripser(self.square_form(),
                    maxdim = maxdim, 
                    thresh = thresh, 
                    coeff = coeff, 
                    distance_matrix = True, 
                    do_cocycles = False)["dgms"]
        i=0
        dic={}
        while i<=maxdim:
            dic["H"+str(i)]=bc(dgms[i])
            i=i+1
        out = pd.Series(dic)
        out.index.name = "homologies"
        return out 

    def bar_code(
            self,
            samplings = "no",
            maxdim = 1,
            thresh = inf,
            coeff = 2,
            print_index = "no"):
        # samplings
        if isinstance(samplings, pd.core.series.Series):
            homologies = pd.Index(["H"+str(d) for d in np.arange(maxdim+1)], 
                                  name = "homologies" )
            square_form = self.square_form()
            Ind = samplings.index
            out = pd.DataFrame(index = Ind, columns = homologies,
                               dtype = object)        
            for i in Ind:
                if print_index == "yes":
                    print(i)
                I = samplings.loc[i]
                if (isinstance(I, str) and 
                    I == "no_sample"):
                    out.loc[i] = bc([[0,0]]) 
                else:
                    ec = distance(square_form[np.ix_(I, I)])
                    out.loc[i] = ec.global_bc(maxdim, thresh, coeff)
            return out.stack().unstack("instances")
        # no samplings
        if (isinstance(samplings, str) and 
              samplings == "no"):
            return self.global_bc(maxdim, thresh, coeff) 
        raise ValueError("""The parmeter samplings has to be either the string "no" or a 
                         result of the samplig method""")
                               


            
class euc_object(object):
    
    def __init__(
            self, 
            points = np.array([[0,0]]), 
            kind = None,  
            name = None):
        points=np.asarray(points, dtype = float)
        self.points=points
        self.dim=points.shape[1] 
        self.kind=kind
        self.name=name
        
    def plot(
            self, 
            color = "No", 
            s = "No"):      
        if self.dim == 2:     
            X = self.points[:, 0]
            Y = self.points[:, 1]
            if color == "No":
                if s == "No":
                    return plt.scatter(X, Y)
                else:
                    return plt.scatter(X, Y, s = s)                    
            else:
                if s == "No":
                    return plt.scatter(X, Y, c = color)
                else:
                    return plt.scatter(X, Y, s = s, c = color)
        elif self.dim == 3:
            fig = plt.figure(self.name)
            ax = fig.add_subplot(111, projection = '3d')
            X = self.points[:, 0]
            Y = self.points[:, 1]
            Z = self.points[:, 2]
            if color == "No": 
                if s == "No":
                    return ax.scatter(X, Y, Z)
                else:
                    return ax.scatter(X, Y, Z, s = s)
            else:
                if s == "No":
                    return ax.scatter(X, Y, Z, c = color)
                else:
                    return ax.scatter(X, Y, Z, s = s, c = color)                   
        else:
            raise ValueError("We can only plot 2 and 3 d objects")
     
    def distance(self, metric = "euclidean"):
        return distance(spatial.distance.pdist(self.points, metric))
    
    def union(
            self,
            other):
        if self.dim == other.dim:
            spoints = self.points
            opoints = other.points

        elif self.dim < other.dim:
            d = other.dim - self.dim
            z = np.zeros((self.number_points, d), dtype=self.points.dtype)            
            spoints = np.concatenate((self.points, z), axis = 1)
            opoints = other.points
        else:
            d = self.dim - other.dim
            z = np.zeros((other.number_points, d), dtype = self.points.dtype)
            opoints = np.concatenate((other.points, z), axis = 1)
            spoints = self.points
        
        allpoints = np.concatenate((spoints, opoints))
        points = np.unique(allpoints, axis = 0)    
        return euc_object(points, self.kind)
    


#################################################
#### CONVERTING BAR CODES INTO STABLE RANKS  ####
#################################################    
def bc_to_sr(barcode):
    # It is assumed that barcode is a bc object
    if isinstance(barcode, bc):
        return barcode.stable_rank()
    # global case no sampling
    elif isinstance(barcode, pd.core.series.Series):
        IH = barcode.index
        out = pd.Series(index = IH, dtype = object)
        for d in IH:
            out.loc[d] = barcode.loc[d].stable_rank()
        return out    
    # sampling 
    elif isinstance(barcode, pd.core.frame.DataFrame):
        Ind = barcode.index
        SR = pd.DataFrame(index = Ind, columns = barcode.columns, 
                          dtype = object)
        out = pd.Series(index = Ind, dtype = object)
        for d in Ind:
            for inst in barcode.columns:
                B = barcode.loc[d].loc[inst]
                SR.loc[d].loc[inst] = B.stable_rank()
            out.loc[d] = np.sum(SR.loc[d]) * (1/ len(barcode.columns)) 
        return out
    else:
        raise ValueError("""The barcode should be either a bc object or an
                          outcome of bar_code method""")       


            


    

    

 
        
