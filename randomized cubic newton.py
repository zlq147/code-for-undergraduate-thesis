# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 15:46:39 2020

@author: hhh
"""

import math #根号，指数等数学函数
import numpy as np
import matplotlib.pyplot as plt
from functions_1 import norm,unit,diag,l1_norm,l2_norm
from time import *
from random import sample #sample用于无重复抽取列表中的元素

#%%
class random_logistic_regression:
    def __init__(self,raw_data,label,lambda_0=2.0,t=200):
        self.data=raw_data
        self.label=label
        self.L1=(l2_norm(self.data)**2)/(4*raw_data.shape[0])+lambda_0
        # self.L2=(l2_norm(self.data)**2)*l1_norm(self.data)/(6*math.sqrt(3)*raw_data.shape[0])
        self.H=1/(6*math.sqrt(3))
        self.alpha=0.25
        self.beta=0.75
        self.eps=1.0e-5
        self.lambda_0=lambda_0
        self.t=t
        
    def logit_(self,w:np.array):
        '''
        目标：计算logit函数,返回一个标量
        '''
        a=np.dot(self.data,w)
        sum_=0
        for i in range(a.shape[0]):
            sum_+=(math.log(1+math.exp(a[i]))-a[i]*self.label[i])
        return sum_/a.shape[0]+self.lambda_0*np.dot(np.transpose(w),w)/2
    
    #logit函数对a=Bw的一阶导数，用g_logit表示
    def g_logit_(self,a:np.array):
        '''
        目标：计算向量logit函数一阶偏导，y为其中的线性部分
        a，y均为列向量
        输出为列向量
        
        对于列向量数组a，索引时要用a[i]
        对于f（a），若不想在函数运算后改变a的值，就不应该在函数内改变a的值
        '''
        g=np.zeros(a.shape)
        for i in range(len(g)):
            g[i]=math.exp(a[i])/(1+math.exp(a[i]))-self.label[i]
        return g/a.shape[0]
    
    #logit函数对w的一阶导数，用p_logit表示
    def p_logit(self,a:np.array,w:np.array,B:np.array):
        
        return np.dot(np.transpose(B),self.g_logit_(a))+self.lambda_0*w
    
    #logit函数对a=Bw的二阶导数，用D_logit表示
    def D_logit(self,a:np.array):
        '''
        目标：计算向量logit函数二阶偏导，y为其中的线性部分
        a，y均为列向量
        输出为列向量
        
        对于列向量数组a，索引时要用a[i]
        对于f（a），若不想在函数运算后改变a的值，就不应该在函数内改变a的值
        '''
        d=np.zeros(a.shape)
        for i in range(len(d)):
            d[i]=math.exp(a[i])/(1+math.exp(a[i]))**2
        return diag(d)/a.shape[0]
    
    #计算b=-B^(T)g-lambda*w
    def b_(self,a:np.array,w:np.array,B:np.array):
        return -self.lambda_0*w-np.dot(np.transpose(B),self.g_logit_(a))
    
    def Z_inv(self,a:np.array,B:np.array,tao:float):
        E1=self.D_logit(a)+tao*self.H/2*np.eye(a.shape[0])
        E2=np.dot(np.transpose(B),np.dot(E1,B))
        E3=self.lambda_0*np.eye(B.shape[1])+E2
        return np.linalg.inv(E3)
    
    #非线性方程||BZ(tao)^(-1)b||-tao
    def res(self,a:np.array,w:np.array,B:np.array,tao:float):
        norm_=norm(np.dot(B,np.dot(self.Z_inv(a,B,tao),self.b_(a,w,B))))
        return norm_-tao
    
    def p_res(self,a:np.array,w:np.array,B:np.array,tao:float):
        e1=np.dot(B,np.dot(self.Z_inv(a,B,tao),self.b_(a,w,B)))
        E4=-self.H/2*np.dot(np.dot(B,self.Z_inv(a,B,tao)),np.transpose(B))
        return np.dot(np.dot(np.transpose(e1),E4),unit(e1))-1
    
    def solve(self,a:np.array,w:np.array,B:np.array):
        '''
        解非线性方程norm（Z（t）^(-1)*g）=t
        '''
        tao=0.5
        stepsize=1
        j=0
        while abs(self.res(a,w,B,tao))>=1.0e-8 :
            tao_delta=stepsize*self.res(a,w,B,tao)/self.p_res(a,w,B,tao)
            while (self.res(a,w,B,tao)**2-self.res(a,w,B,tao-tao_delta)**2)<=2*self.alpha*stepsize*self.res(a,w,B,tao)**2:
                stepsize*=self.beta
                tao_delta*=self.beta
            tao-=tao_delta
            j+=1
        return np.dot(self.Z_inv(a,B,tao),self.b_(a,w,B))
    
    def cubic_newton_train(self):
        B=self.data
        w=np.zeros([B.shape[1],1])
        a=np.zeros([B.shape[0],1])
        stop=False
        length=50
        k=0
        obj_values=[self.logit_(w)]
        eps_list=[]
        while not stop:
            sampled_nums=sample(nums,self.t) #sample进行无重复抽样
            B1=np.empty([0,d]) #B的抽样
            w1=np.empty([0,1]) #w的抽样
            #np.concatenate表示矩阵拼接
            #B【：，num】,w[num]为一维数组，而空数组B1为二维数组，因此需要将B【：，num】,w[num]转换为二维数组
            #利用一维数组.reshape（shape）方法将一维数组转换为二维数组
            for num in sampled_nums:
                B1=np.concatenate((B1,B[:,num].reshape((1,d))),axis=0) #a[:,i]为数组a的第i列,但索引后仅得到行向量
                w1=np.concatenate((w1,w[num].reshape((1,1))),axis=0)
            B1=np.transpose(B1) #合并后进行转置
            w_delta=self.solve(a,w1,B1)
            a_delta=np.dot(B1,w_delta)
            for i in range(len(sampled_nums)):
                w[sampled_nums[i]]+=w_delta[i]
            a+=a_delta
            k+=1
            obj_values.append(self.logit_(w))
            eps_list.append(obj_values[-2]-obj_values[-1])
            if len(obj_values)>=(length+1):
                del(eps_list[0])
                if max(eps_list)<=self.eps:
                    stop=True
            print(k,"th value=",self.logit_(w))
        
        return w,obj_values,k
        
    def grad_train(self):
        B=self.data
        w=np.zeros([B.shape[1],1])
        a=np.zeros([B.shape[0],1])
        stop=False
        length=50
        k=0
        obj_values=[self.logit_(w)]
        eps_list=[]
        while not stop:
            sampled_nums=sample(nums,self.t) #sample进行无重复抽样
            B1=np.empty([0,d]) #B的抽样
            w1=np.empty([0,1]) #w的抽样
            #np.concatenate表示矩阵拼接
            #B【：，num】,w[num]为一维数组，而空数组B1为二维数组，因此需要将B【：，num】,w[num]转换为二维数组
            #利用一维数组.reshape（shape）方法将一维数组转换为二维数组
            for num in sampled_nums:
                B1=np.concatenate((B1,B[:,num].reshape((1,d))),axis=0) #a[:,i]为数组a的第i列,但索引后仅得到行向量
                w1=np.concatenate((w1,w[num].reshape((1,1))),axis=0)
            B1=np.transpose(B1) #合并后进行转置
            w_delta=-self.p_logit(a,w1,B1)/self.L1
            a_delta=np.dot(B1,w_delta)
            for i in range(len(sampled_nums)):
                w[sampled_nums[i]]+=w_delta[i]
            a+=a_delta
            k+=1
            obj_values.append(self.logit_(w))
            eps_list.append(obj_values[-2]-obj_values[-1])
            if len(obj_values)>=(length+1):
                del(eps_list[0])
                if max(eps_list)<=self.eps:
                    stop=True
            print(k,"th value=",self.logit_(w))
        
        plt.scatter([i for i in range(len(obj_values))],obj_values)
        plt.title("eps=1.0e-5")
        plt.legend(["randomized cubic newton","randomized block gradient"])
        return w,obj_values,k
   
        
#%%
datas=[]
#read the txt data
with open('D:\毕设\毕设程序\leu.txt') as file_object: 
    '''
    np.array赋值时应加双层【】号，否则有维数问题
    '''
    for line in file_object:
        #nonetype after removing,cannot be applied directly
        data={}
        data_str=line.split(" ")
        data_str.remove("")
        #convert the string to float
        data["y"]=np.array([[float(data_str[0])]])
        data["x"]=[]
        data_str.pop(0)
        for element in data_str:
            data["x"].append(float(element.split(":")[-1]))
        data["x"]=np.array([data["x"]])
        datas.append(data)

n=datas[0]["x"].shape[1] #数据维数
d=len(datas) #数据个数
nums=[i for i in range(n)]
B=np.empty([0,n])
y=np.empty([0,1]) #labels
#数据矩阵,B为原矩阵，下面的B1为抽样的矩阵
for data in datas:
    B=np.concatenate((B,data["x"]),axis=0)
    y=np.concatenate((y,(data["y"]+1)/2),axis=0) #将l的负标签转换成0标签  

#%%
model=random_logistic_regression(B,y,lambda_0=1.0)

begin_time1=time()
w1,obj_values1,k1=model.cubic_newton_train()
end_time1=time()
run_time1=end_time1-begin_time1

begin_time2=time()
w2,obj_values2,k2=model.grad_train()
end_time2=time()
run_time2=end_time2-begin_time2

plt.scatter([i for i in range(len(obj_values1))],obj_values1,c='r',label='randomized cubic newton')        
plt.scatter([i for i in range(len(obj_values2))],obj_values2,c='b',label='randomized block gradient')
plt.title("lambda=1.0")
plt.legend()

print("time of cubic newton:",run_time1)
print("iterations of cubic newton:",k1)
print("time of grad:",run_time2)
print("iterations of grad:",k2)

        
