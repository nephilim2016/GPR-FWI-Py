#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 13:46:58 2018

@author: nephilim
"""
import numpy as np
import shutil
import os

class para():
    def __init__(self):
        pass
    
class options():
    def __init__(self):
        pass

class Optimization(para,options):
    def __init__(self,fh,iepsilon):
        super().__init__()
        self.fh=fh
        self.data=iepsilon.flatten()
        
    def optimization(self):
        # Set default values for options if they don't exist
        options.maxiter = getattr(options, 'maxiter', 10)
        options.maxit_ls = getattr(options, 'maxit_ls', 5)
        options.M = getattr(options, 'M', 5)
        options.tol = getattr(options, 'tol', 1e-2)
        options.c1 = getattr(options, 'c1', 1e-4)
        options.c2 = getattr(options, 'c2', 0.9)
        options.ls_int = getattr(options, 'ls_int', 2)
        options.progTol = getattr(options, 'progTol', 1e-9)
        
        # Prepare the directory for saving models
        model_dir = f'./{para.target_freq}Hz_imodel_file_{para.Source_num_select}_{para.change_source}'
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        os.makedirs(model_dir)
        
        n = len(self.data)
        d = np.zeros(n)
        x = self.data.copy()
        iter_ = 0
        alpha0 = 1.0
        alpha = 1.0
        # Print header for the optimization progress table
        print(f'{"iter":>5s}, {"eval":>6s}, {"step length":>15s}, {"function value":>15s}, {"||g(x)||_2":>15s}\n')
        
        # Compute initial function and gradient
        f, g = self.fh(x.reshape((para.xl + 20, -1)))
        g_old = g
        f0 = f
        f_old=f
        fevals = 1
        info = [[iter_, fevals, alpha0, f, np.linalg.norm(g, 2)]]
        
        # Print initial values
        print(f'{iter_:5d}, {fevals:6d}, {alpha0:15.5e}, {f:15.5e}, {np.linalg.norm(g, 2):15.5e}\n')
        
        for iter_ in range(options.maxiter):
            if iter_ == 0:
                S = np.zeros((n, options.M))
                Y = np.zeros((n, options.M))
            else:
                S = np.hstack((S[:, 1:], S[:, 0][:, np.newaxis]))
                Y = np.hstack((Y[:, 1:], Y[:, 0][:, np.newaxis]))
                S = np.hstack((S[:, :-1], alpha * d[:, np.newaxis]))
                Y = np.hstack((Y[:, :-1], (g - g_old)[:, np.newaxis]))

            # Update direction and handle negative curvature
            d = self.__B(-g, S, Y, np.array([]))
            p = -np.dot(d, g) / np.dot(g, g)
            
            if p <= 0:
                S = np.zeros((n, options.M))
                Y = np.zeros((n, options.M))
            
            g_old = g
            gtd = np.dot(g, d)

            if iter_ == 0:
                alpha0 = -f / gtd
                alpha0 = max(1e-5, min(10, alpha0))
            else:
                # alpha0 = 1
                alpha0 = min(1, 2*(f-f_old)/(gtd))
                alpha0 = max(1e-5, min(0.15, alpha0))
                
            f_old, gtd_old = f, gtd
            
            # Perform Wolfe line search
            alpha, f, g, lsiter = self.__WolfeLineSearch(x, alpha0, d, f, g, gtd, options.c1, options.c2, options.ls_int, options.maxit_ls, self.fh)
            fevals += lsiter
            x += alpha * d
            x = self.__counts(x)
            
            # Save current model
            np.save(f'./{para.target_freq}Hz_imodel_file_{para.Source_num_select}_{para.change_source}/{iter_}_imodel.npy', x)
            
            # Record progress
            info.append([iter_, fevals, alpha, f, np.linalg.norm(g, 2)])
            
            # Print progress
            print(f'{iter_ + 1:5d}, {fevals:6d}, {alpha:15.5e}, {f:15.5e}, {np.linalg.norm(g, 2):15.5e}\n')
            
            # Check stopping conditions
            if f / f0 < options.tol:
                print('Function Value less than funTol\n')
                break
            if alpha == 0:
                print('Alpha Value equal to 0\n')
                break
            if fevals >= 1000:
                print('Reached Maximum Number of Function Evaluations\n')
                break
            if iter_ == options.maxiter:
                print('Reached Maximum Number of Iterations\n')
                break

        return x, info
    
    def __B(self, x, S, Y, H0):
        # Select non-zero columns of S
        J = np.sum(abs(S), axis=0).nonzero()[0]
        S = S[:, J]
        Y = Y[:, J]
        M = S.shape[1]
        n = len(x)

        # Initialize H0 if it's empty and M > 0
        if (H0.size == 0) and (M > 0):
            H0 = np.linalg.norm(Y[:, -1], 2)**2 / np.dot(S[:, -1], Y[:, -1]) * np.ones(n)
        else:
            H0 = np.ones(n)

        # Initialize alpha and rho
        alpha = np.zeros(M)
        rho = np.zeros(M)
        
        # Compute rho values
        for k in range(M):
            rho[k] = 1 / np.dot(Y[:, k], S[:, k])
        
        # First loop: compute q
        q = x.copy()
        for k in range(M-1, -1, -1):
            alpha[k] = rho[k] * np.dot(S[:, k], q)
            q -= alpha[k] * Y[:, k]

        # Compute z
        z = q / H0
        
        # Second loop: compute z using alpha and rho
        for k in range(M):
            beta = rho[k] * np.dot(Y[:, k], z)
            z += (alpha[k] - beta) * S[:, k]

        return z

    
    def __WolfeLineSearch(self, x, t, d, f, g, gtd, c1, c2, LS_interp, maxLS, fh):
        # Initialize new values
        x_new = x + t * d
        x_new = self.__counts(x_new)
        
        f_new, g_new = fh(x_new.reshape((para.xl + 20, -1)))
        
        lsiter = 1
        gtd_new = np.dot(g_new, d)
        LSiter = 0
        t_prev = 0.0
        f_prev, g_prev, gtd_prev = f, g, gtd
        done = 0

        while LSiter < maxLS:
            # Armijo condition
            if (f_new > f + c1 * t * gtd) or (LSiter > 1 and f_new >= f_prev):
                bracket = np.hstack((t_prev, t))
                bracketFval = np.hstack((f_prev, f_new))
                bracketGval = np.hstack((g_prev[:, np.newaxis], g_new[:, np.newaxis]))
                break
            elif abs(gtd_new) <= -c2 * gtd:
                bracket, bracketFval, bracketGval, done = t, f_new, g_new, 1
                break
            elif gtd_new >= 0:
                bracket = np.hstack((t_prev, t))
                bracketFval = np.hstack((f_prev, f_new))
                bracketGval = np.hstack((g_prev[:, np.newaxis], g_new[:, np.newaxis]))
                break

            # Update step size
            t_prev, f_prev, g_prev, gtd_prev = t, f_new, g_new, gtd_new
            minStep = t + 0.01 * (t - t_prev)
            maxStep = t * 10
            t = maxStep if LS_interp <= 1 else self.__polyinterp(
                np.array([[t_prev, f_prev, gtd_prev], [t, f_new, gtd_new]]),
                minStep, maxStep
            )

            # Compute new function and gradient values
            x_new = self.__counts(x + t * d)
            f_new, g_new = fh(x_new.reshape((para.xl + 20, -1)))

            lsiter += 1
            gtd_new = np.dot(g_new, d)
            LSiter += 1

        # If max line search iterations reached
        if LSiter == maxLS:
            bracket = np.hstack((0, t))
            bracketFval = np.hstack((f, f_new))
            bracketGval = np.hstack((g_prev[:, np.newaxis], g_new[:, np.newaxis]))

        insufProgress = 0
        while not done and LSiter < maxLS:
            LOpos = np.argmin(bracketFval)
            f_LO = bracketFval[LOpos]
            HIpos = -LOpos + 1

            # Interpolation step
            if LS_interp <= 1:
                t = np.mean(bracket)
            else:
                t = self.__polyinterp(
                    np.array([[bracket[0], bracketFval[0], np.dot(bracketGval[:, 0], d)],
                              [bracket[1], bracketFval[1], np.dot(bracketGval[:, 1], d)]]))
                print('Lines Search Grad-Cubic Interpolation Iteration',LSiter,'alpha=',t)
            # Adjust step if insufficient progress
            if np.min((np.max(bracket) - t, t - np.min(bracket))) / (np.max(bracket) - np.min(bracket)) < 0.1:
                if insufProgress or t >= np.max(bracket) or t <= np.min(bracket):
                    t = np.max(bracket) - 0.1 * (np.max(bracket) - np.min(bracket)) \
                        if abs(t - np.max(bracket)) < abs(t - np.min(bracket)) \
                        else np.min(bracket) + 0.1 * (np.max(bracket) - np.min(bracket))
                    insufProgress = 0
                else:
                    insufProgress = 1
            else:
                insufProgress = 0

            # Update function and gradient values
            x_new = self.__counts(x + t * d)
            f_new, g_new = fh(x_new.reshape((para.xl + 20, -1)))
            lsiter += 1
            gtd_new = np.dot(g_new, d)
            LSiter += 1

            # Check conditions
            armijo = f_new < f + c1 * t * gtd
            if not armijo or f_new >= f_LO:
                bracket[HIpos], bracketFval[HIpos], bracketGval[:, HIpos] = t, f_new, g_new
            else:
                if abs(gtd_new) <= -c2 * gtd:
                    done = 1
                elif gtd_new * (bracket[HIpos] - bracket[LOpos]) >= 0:
                    bracket[HIpos], bracketFval[HIpos], bracketGval[:, HIpos] = bracket[LOpos], bracketFval[LOpos], bracketGval[:, LOpos]
                bracket[LOpos], bracketFval[LOpos], bracketGval[:, LOpos] = t, f_new, g_new

        # Final values
        if isinstance(bracket, np.ndarray):
            LOpos = np.argmin(bracketFval)
            t, f_new, g_new = bracket[LOpos], bracketFval[LOpos], bracketGval[:, LOpos]
        else:
            t, f_new, g_new = bracket, bracketFval, bracketGval

        return t, f_new, g_new, lsiter

    
    def __polyinterp(self, points, *vargs):
        # Set default bounds and plotting flag
        xmin = np.min(points[:, 0])
        xmax = np.max(points[:, 0])
        
        xminBound = vargs[0] if len(vargs) >= 1 else xmin
        xmaxBound = vargs[1] if len(vargs) >= 2 else xmax

        # Determine minimum position
        minPos = np.argmin(points[:, 0])
        notMinPos = -minPos + 1

        # Handle edge case where the points have the same x value
        if (points[minPos, 0] - points[notMinPos, 0]) == 0:
            return (xmaxBound + xminBound) / 2

        # Calculate intermediate values
        d1 = (points[minPos, 2] + points[notMinPos, 2] -
              3 * (points[minPos, 1] - points[notMinPos, 1]) /
              (points[minPos, 0] - points[notMinPos, 0]))
        
        d2 = np.sqrt(d1**2 - points[minPos, 2] * points[notMinPos, 2])

        # If d2 is real, calculate the interpolation
        if d2.imag == 0.0:
            t = points[notMinPos, 0] - (points[notMinPos, 0] - points[minPos, 0]) * \
                ((points[notMinPos, 2] + d2.real - d1) /
                 (points[notMinPos, 2] - points[minPos, 2] + 2 * d2.real))

            minPos = np.clip(t, xminBound, xmaxBound)
        else:
            # If d2 is not real, return the midpoint
            minPos = (xmaxBound + xminBound) / 2

        return minPos

        
    def __counts(self, x):
        x = np.clip(x, 1, 81)
        return x
