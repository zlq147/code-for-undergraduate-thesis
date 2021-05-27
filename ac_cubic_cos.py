# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 18:41:28 2021

@author: hhh
"""
#加载相应包
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import uniform
from time import time
from functions_1 import norm,unit,diag,l1_norm,l2_norm

#%%
class cubic_min:
    def __init__(self,initial,lambda_0=1.0,eps=1.0e-8):
        #问题维数
        self.initial=initial
        #问题维数
        self.dim=initial.shape[0]
        #一阶Lipschitz常数L1
        self.lambda_0=lambda_0
        self.L1=lambda_0*2
        #二阶Lipschitz常数L2
        self.L2=6.0
        self.alpha=0.25
        self.beta=0.75
        self.eps=eps
        self.mu=0.5
        self.M1=self.L1/self.mu
        self.M2=self.L2/self.mu
        self.N1=self.M1*(1+self.mu)*math.sqrt(2)
        self.N2=self.M2*(1+self.mu)*128/9
        
    def h(self,x:float):
        if x>=0:
            if 0<x<1:
                return 5/36*(x**3)-(x**3)/6*math.log(x)
            elif (x-int(x))==0:
                return 5/36*x+1/4*x*(x-1)/2
            else:
                return self.h(float(int(x)))+1/4*int(x)*(x-int(x))+self.h(x-int(x))
        else:
            return self.h(-1*x)
            
    def p_h(self,x:float):
        if x>=0:
            if 0<x<1:
                return 1/4*(x**2)-(x**2)/2*math.log(x)
            elif (x-int(x))==0:
                return x/4
            else:
                return self.p_h(float(int(x)))+self.p_h(x-int(x))
        else:
            return -self.p_h(-1*x)
        
    def h1(self,x:float):
        return x**2/2-math.cos(x)
    
    def p_h1(self,x:float):
        return x+math.sin(x)
    
    #logit函数
    def logit(self,w:np.array):
        '''
        目标：计算logit函数,返回一个标量
        '''
        # sin_sum=sum([math.sin(w[i]) for i in range(len(w))])
        # sum_=self.L1*(np.dot(np.transpose(w),w)/2+sin_sum)
        sum_=sum([self.lambda_0*self.h1(w[i])+abs(w[i]**3) for i in range(len(w))])
        return sum_
              
    #logit函数一阶导数
    def p_logit(self,w:np.array):
        '''
        目标：计算向量logit函数一阶偏导，y为其中的线性部分
        a，y均为列向量
        输出为列向量
        
        对于列向量数组a，索引时要用a[i]
        对于f（a），若不想在函数运算后改变a的值，就不应该在函数内改变a的值
        '''
        g=np.zeros(w.shape)
        for i in range(len(w)):
            g[i]=self.lambda_0*self.p_h1(w[i])+3*w[i]*abs(w[i])
        #print("type(g)=",type(g))
        return g
    
    #logit函数二阶导数
    def p_p_logit(self,w:np.array):
        '''
        目标：计算向量logit函数二阶海瑟矩阵对角元构成的列向量（由于logit（a）具有可分离性）
        a为列向量
        输出为列向量
        '''
        b=np.zeros(w.shape)
        for i in range(len(w)):
            b[i]=6*abs(w[i])
        H=diag(b)
        return H
    
    #非线性方程组矩阵的逆
    def Z_inverse(self,w:np.array,tao:float):
        '''
        目标：计算Z（t）^(-1)
        '''
        return np.linalg.inv(np.eye(w.shape[0])*(2*self.L1+self.L2*tao/2)+self.p_p_logit(w))
    
    #非线性方程
    def res(self,w:np.array,tao:float):
        '''
        目标：计算norm（Z（t）^(-1)*g）-t
        '''
        return norm(np.dot(self.Z_inverse(w,tao),self.p_logit(w)))-tao
    
    #非线性方程导数
    def p_res(self,w:np.array,tao:float):
        '''
        目标：计算f(t)=norm（Z（t）^(-1)*g）-t的导数
        '''
        g1=np.dot(self.Z_inverse(w,tao),self.p_logit(w))
        y1=-self.L2*np.dot(np.transpose(unit(g1)),np.dot(self.Z_inverse(w,tao),g1))/2
        return y1-1
    
    #解非线性方程
    def solve(self,w:np.array):
        '''
        解非线性方程norm（Z（t）^(-1)*g）=t
        '''
        tao=0
        stepsize=1
        j=0
        while abs(self.res(w,tao))>=1.0e-8 :
            tao_delta=stepsize*self.res(w,tao)/self.p_res(w,tao)
            while (self.res(w,tao)**2-self.res(w,tao-tao_delta)**2)<=2*self.alpha*stepsize*self.res(w,tao)**2:
                stepsize*=self.beta
                tao_delta*=self.beta
            tao-=tao_delta
            j+=1
        return -np.dot(self.Z_inverse(w,tao),self.p_logit(w))
    
    #训练
    def train(self):
        '''
        参数更新(自己定义的函数名不能被当作变量名)
        '''
        w=self.initial
        delta=1.0
        k=1
        obj_values=[self.logit(w)]
        while delta>=self.eps:
            print(k)
            print("norm=",norm(self.p_logit(w)))
            w+=self.solve(w)
            obj_values.append(self.logit(w))
            k+=1
            delta=norm(self.p_logit(w))  
            
        plt.scatter([i for i in range(len(obj_values))],obj_values)
        return w,obj_values,norm(self.p_logit(w)),k
    
    #非线性方程组矩阵的逆
    def Z_inv_1(self,w:np.array,tao:float):
        '''
        目标：计算Z（t）^(-1)
        '''
        return np.linalg.inv(np.eye(w.shape[0])*(self.M1+self.M2*tao/2)+self.p_p_logit(w))
    
    #非线性方程
    def res1(self,w:np.array,tao:float):
        '''
        目标：计算norm（Z（t）^(-1)*g）-t
        '''
        return norm(np.dot(self.Z_inv_1(w,tao),self.p_logit(w)))-tao
    
    #非线性方程导数
    def p_res1(self,w:np.array,tao:float):
        '''
        目标：计算f(t)=norm（Z（t）^(-1)*g）-t的导数
        '''
        g1=np.dot(self.Z_inv_1(w,tao),self.p_logit(w))
        y1=-self.M2*np.dot(np.transpose(unit(g1)),np.dot(self.Z_inv_1(w,tao),g1))/2
        return y1-1
    
    #解非线性方程
    def solve1(self,w:np.array):
        '''
        解非线性方程norm（Z（t）^(-1)*g）=t
        '''
        tao=0
        stepsize=1
        j=0
        while abs(self.res1(w,tao))>=1.0e-8 :
            tao_delta=stepsize*self.res1(w,tao)/self.p_res1(w,tao)
            while (self.res1(w,tao)**2-self.res1(w,tao-tao_delta)**2)<=2*self.alpha*stepsize*self.res1(w,tao)**2:
                stepsize*=self.beta
                tao_delta*=self.beta
            tao-=tao_delta
            j+=1
        return -np.dot(self.Z_inv_1(w,tao),self.p_logit(w))
    
    def R(self,w:np.array,w0:np.array):
        return (self.L1+self.M1+self.N1)*norm(w-w0)**2/2+(self.L2+self.M2+self.N2)*norm(w-w0)**3/6
    
    #训练
    def ac_train(self):
        '''
        参数更新(自己定义的函数名不能被当作变量名)
        '''
        w0=self.initial
        delta=1.0
        k=1
        w=w0+self.solve(w0)
        obj_values=[self.logit(w)]
        l=np.zeros(w0.shape)
        A_=1
        while k<=1000:#delta>=self.eps:
            print(k)
            print("norm=",norm(self.p_logit(w)))
            a_=k+1
            A_+=a_
            alpha=a_/A_
            v=w0-l/((self.N1+math.sqrt(self.N1**2+2*self.N2*norm(l)))/2)
            v=alpha*v+(1-alpha)*w
            w=v+self.solve1(v)
            # obj_values.append((self.logit(w)+self.dim)*k**2)
            obj_values.append((self.logit(w)+self.lambda_0*self.dim)*k**2)
            l+=a_*self.p_logit(w)
            k+=1
            delta=norm(self.p_logit(w))
            
        # plt.scatter([i for i in range(len(obj_values))],obj_values)
        return w,obj_values,norm(self.p_logit(w)),k
    
#%%
w0=np.zeros([4,1])
for j in range(4):
    w0[j]=uniform(-5,5)
model=cubic_min(w0,2.0)
w,obj_values,norm,k=model.ac_train()
plt.scatter([l for l in range(k)],obj_values)


