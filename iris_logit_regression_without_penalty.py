# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 17:20:00 2021
cubic solving for iris without penalty
@author: hhh
"""

import math
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from time import *
from functions_1 import *

#%%
'''读取数据'''
iris = load_iris() #加载iris数据集
df=pd.DataFrame(iris.data,columns=iris.feature_names) #将数据集填入列表中
df["label"]=iris.target
df1=df[:100]

#%%
class logit_regression_:
    
    def __init__(self,raw_data,eps=1.0e-5):
        self.data=np.array(raw_data)[:,[i for i in range(raw_data.shape[1]-1)]]
        self.label=np.array(raw_data)[:,[-1]]
        self.L1=(l2_norm(self.data)**2)/(4*raw_data.shape[0])
        self.L2=(l2_norm(self.data)**2)*l1_norm(self.data)/(6*math.sqrt(3)*raw_data.shape[0])
        self.M2=2*self.L2
        self.N2=12*self.L2
        self.alpha=0.25
        self.beta=0.75
        self.eps=eps
        
    #logit函数
    def logit_(self,w:np.array):
        '''
        目标：计算logit函数,返回一个标量
        '''
        a=np.dot(self.data,w)
        sum_=0
        for i in range(a.shape[0]):
            sum_+=(math.log(1+math.exp(a[i]))-a[i]*self.label[i])
        return sum_/a.shape[0]
              
    #logit函数一阶导数
    def p_logit_(self,w:np.array):
        '''
        目标：计算向量logit函数一阶偏导，y为其中的线性部分
        a，y均为列向量
        输出为列向量
        
        对于列向量数组a，索引时要用a[i]
        对于f（a），若不想在函数运算后改变a的值，就不应该在函数内改变a的值
        '''
        #print("type(w)=",type(w))
        #print(w)
        a=np.dot(self.data,w)
        b=np.zeros(a.shape)
        #print("type(b)=",type(b))
        for i in range(len(b)):
            b[i]=math.exp(a[i])/(1+math.exp(a[i]))-self.label[i]
        g=np.dot(np.transpose(self.data),b)
        #print("type(g)=",type(g))
        return g/a.shape[0]
    
    #logit函数二阶导数
    def p_p_logit_(self,w:np.array):
        '''
        目标：计算向量logit函数二阶海瑟矩阵对角元构成的列向量（由于logit（a）具有可分离性）
        a为列向量
        输出为列向量
        '''
        a=np.dot(self.data,w)
        b=np.zeros(a.shape)
        for i in range(len(a)):
            b[i]=math.exp(a[i])/((1+math.exp(a[i]))**2)
        H=np.dot(np.transpose(self.data),np.dot(diag(b),self.data))
        return H/a.shape[0]
    
    #非线性方程组矩阵的逆
    def Z_inverse_(self,w:np.array,tao:float):
        '''
        目标：计算Z（t）^(-1)
        '''
        return np.linalg.inv(np.eye(w.shape[0])*self.L2*tao/2+self.p_p_logit_(w))
    
    #非线性方程
    def res_(self,w:np.array,tao:float):
        '''
        目标：计算norm（Z（t）^(-1)*g）-t
        '''
        return norm(np.dot(self.Z_inverse_(w,tao),self.p_logit_(w)))-tao
    
    #非线性方程导数
    def p_res_(self,w:np.array,tao:float):
        '''
        目标：计算f(t)=norm（Z（t）^(-1)*g）-t的导数
        '''
        g1=np.dot(self.Z_inverse_(w,tao),self.p_logit_(w))
        y1=-self.L2*np.dot(np.transpose(unit(g1)),np.dot(self.Z_inverse_(w,tao),g1))/2
        return y1-1
    
    #解非线性方程
    def solve_(self,w:np.array):
        '''
        解非线性方程norm（Z（t）^(-1)*g）=t
        '''
        tao=1
        stepsize=1
        j=0
        while abs(self.res_(w,tao))>=1.0e-8 :
            tao_delta=stepsize*self.res_(w,tao)/self.p_res_(w,tao)
            while (self.res_(w,tao)**2-self.res_(w,tao-tao_delta)**2)<=2*self.alpha*stepsize*self.res_(w,tao)**2:
                stepsize*=self.beta
                tao_delta*=self.beta
            tao-=tao_delta
            j+=1
        return -np.dot(self.Z_inverse_(w,tao),self.p_logit_(w))
    
    #训练
    def train_(self):
        '''
        参数更新(自己定义的函数名不能被当作变量名)
        '''
        w=np.zeros([self.data.shape[1],1])
        delta=1.0
        k=1
        obj_values=[self.logit_(w)]
        while delta>=self.eps:
            w+=self.solve_(w)
            obj_values.append(self.logit_(w))
            k+=1
            delta=norm(self.p_logit_(w))
        
        return w,obj_values,norm(self.p_logit_(w)),k
    
    #非线性方程组矩阵的逆
    def Z_inv_1(self,w:np.array,tao:float):
        '''
        目标：计算Z（t）^(-1)
        '''
        return np.linalg.inv(np.eye(w.shape[0])*self.M2*tao/2+self.p_p_logit_(w))
    
    #非线性方程
    def res1(self,w:np.array,tao:float):
        '''
        目标：计算norm（Z（t）^(-1)*g）-t
        '''
        return norm(np.dot(self.Z_inv_1(w,tao),self.p_logit_(w)))-tao
    
    #非线性方程导数
    def p_res1(self,w:np.array,tao:float):
        '''
        目标：计算f(t)=norm（Z（t）^(-1)*g）-t的导数
        '''
        g1=np.dot(self.Z_inv_1(w,tao),self.p_logit_(w))
        y1=-self.M2*np.dot(np.transpose(unit(g1)),np.dot(self.Z_inv_1(w,tao),g1))/2
        return y1-1
    
    #解非线性方程
    def solve1(self,w:np.array):
        '''
        解非线性方程norm（Z（t）^(-1)*g）=t
        '''
        tao=0.5
        stepsize=1
        j=0
        while abs(self.res1(w,tao))>=1.0e-8 :
            tao_delta=stepsize*self.res1(w,tao)/self.p_res1(w,tao)
            while (self.res1(w,tao)**2-self.res1(w,tao-tao_delta)**2)<=2*self.alpha*stepsize*self.res1(w,tao)**2:
                stepsize*=self.beta
                tao_delta*=self.beta
            tao-=tao_delta
            j+=1
        return -np.dot(self.Z_inv_1(w,tao),self.p_logit_(w))
 
    def ac_train_(self):
        '''
        参数更新(自己定义的函数名不能被当作变量名)；
        对于二维数组，若把变量赋给变量，两个变量
        可能同时变化。
        若想给二维数组一个不随其他变量变化的值，
        尽量不通过变量直接赋值。
        
        w0=np.zeros([4,1]);
        print(w0);
        l=w0;
        l+=np.ones([4,1]);
        print(w0)
        此时w0与l同时变化
        '''
        w0=np.zeros([self.data.shape[1],1])
        delta=1.0
        k=1
        w=w0+self.solve_(w0)
        obj_values=[self.logit_(w)]
        l=np.zeros(w0.shape) #正招
        #l=w0 #败招
        A_=1
        while delta>=self.eps: 
            a_=(k+1)*(k+2)/2
            A_+=a_
            alpha=a_/A_
            v=w0-math.sqrt(2/self.N2)*unit(l)*math.sqrt(norm(l))
            v=alpha*v+(1-alpha)*w
            w=v+self.solve1(v)
            obj_values.append(self.logit_(w))
            l+=a_*self.p_logit_(w)
            k+=1
            delta=norm(self.p_logit_(w))
         
        return w,obj_values,norm(self.p_logit_(w)),k 
    
    def newton_train_1(self):
        w=np.zeros([self.data.shape[1],1])
        delta=1.0
        k=1
        obj_values=[self.logit_(w)]
        while delta>=self.eps: 
            step=1.0
            w_delta=np.dot(np.linalg.inv(self.p_p_logit_(w)),self.p_logit_(w))
            stop=(self.logit_(w-step*w_delta)-self.logit_(w)<=-self.alpha*step*np.dot(np.transpose(self.p_logit_(w)),w_delta))
            while not stop :
                step*=self.beta
                stop=(self.logit_(w-step*w_delta)-self.logit_(w)<=-self.alpha*step*np.dot(np.transpose(self.p_logit_(w)),w_delta))
            w-=step*w_delta
            k+=1
            delta=norm(self.p_logit_(w))
            obj_values.append(self.logit_(w))
        print("iter=",k)
            
        plt.scatter([i for i in range(len(obj_values))],obj_values)
        return w,obj_values,norm(self.p_logit_(w))
    
    def newton_train_2(self):
        w=np.zeros([self.data.shape[1],1])
        delta=1.0
        k=1
        obj_values=[self.logit_(w)]
        while delta>=self.eps: 
            w_delta=np.dot(np.linalg.inv(self.p_p_logit_(w)),self.p_logit_(w))
            w-=w_delta
            k+=1
            delta=norm(self.p_logit_(w))
            obj_values.append(self.logit_(w))
        print("iter=",k)
            
        plt.scatter([i for i in range(len(obj_values))],obj_values)
        return w,obj_values,norm(self.p_logit_(w))
    
    def grad_train_(self):
        w=np.zeros([self.data.shape[1],1])
        delta=1.0
        k=1
        obj_values=[self.logit_(w)]
        while delta>=self.eps:
            w-=self.p_logit_(w)/self.L1
            k+=1
            delta=norm(self.p_logit_(w))
            obj_values.append(self.logit_(w))
            print("k=",k)
            print("norm=",delta)
        print("iter=",k)
        return w,obj_values,norm(self.p_logit_(w))
        

#%%
model=logit_regression_(df1,eps=1.0e-3)
begin_time1=time()
w1,obj_values1,norm_1,k=model.train_()
end_time1=time()
run_time1=end_time1-begin_time1

begin_time2=time()
w2,obj_values2,norm_2,k=model.ac_train_()
end_time2=time()
run_time2=end_time2-begin_time2

plt.subplot(1,2,1)
plt.scatter([i for i in range(len(obj_values1))],obj_values1)
plt.title("eps=1.0e-3")
plt.legend(["no acceleration"])

plt.subplot(1,2,2)
plt.scatter([i for i in range(len(obj_values2))],obj_values2)
plt.title("eps=1.0e-3")
plt.legend(["accelerated"])


print("time for no acceleration:",run_time1)
print("time for acceleration:",run_time2)

