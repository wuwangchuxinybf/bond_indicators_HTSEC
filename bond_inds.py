# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 10:15:17 2019

@author: wuwangchuxin
"""

#fixed income indicators calculating
import numpy as np
import datetime 
import QuantLib as ql
#import sympy as sy
import math
from scipy.optimize import fsolve

# 输入
# 债券名称、起息日、票面利率，票面利率，当前余额
# 债券基本信息

#付息日矩阵 面值矩阵  利率矩阵 生成未来现金流 （利息加本金）
#未来

class Bond_Profile:
    def __init__(self,code,faceAmount,cleanPrice,valueDate,maturity,couponRate,frequency,
                 bondType,hasopions,exchange,standard,**kw):
        self.code = code #债券代码
        self.faceAmount = faceAmount #面值
        self.cleanPrice = cleanPrice #净价 #行情软件一般得到的都是债券净价报价
#        self.full_price = full_price #全价
#        self.issueDate = issueDate #发行日期
        self.valueDate = valueDate #计息日期
        self.maturity = maturity #到期日期
        self.couponRate = couponRate #票面利率
        self.frequency = frequency #年付息频率(1 Annual 2 Semiannual 4 Quarterly 12 Monthly)
#        self.interestType = interestType #利率类型（1单利，2按年复利，暂不考虑连续复利）
        #付息类型（1固定利息按实际天数付息，2固定利息按平均值付息，3浮动利息,4到期一次还本付息,5零息债券、贴现债）
        self.bondType = bondType 
        self.hasopions = hasopions #是否含权债
        self.exchange = exchange #交易市场：IB银行间 SSE交易所
        self.standard = standard #计算基准：中债
        self.nowdate = datetime.datetime.now().strftime('%Y-%m-%d') #d当前日期
        for info in kw:
            setattr(self,info,kw[info])
    
    @staticmethod
    def format_qldate(datestr):
        '''将字符串日期转换为quantlib类型的日期'''
        return ql.Date(*[int(x) for x in datestr.split('-')[::-1]])
    
    def schedule(self):
        '''返回付息日期序列'''
        s_date = self.format_qldate(self.valueDate)
        edate = self.format_qldate(self.maturity)
        freq = ql.Period(self.frequency) #ql.Annual
        if self.exchange=='IB':
            calendar = ql.China(ql.China.IB)#时间标准 ql.China(),银行间
        elif self.exchange=='SSE':
            calendar = ql.China(ql.China.SSE)#交易所
        schedule = ql.Schedule(s_date, edate, freq, calendar, ql.Following,
                                   ql.Following, ql.DateGeneration.Forward, False)
        return list(schedule)
    
    def info_matrix(self):
        '''计算基础指标，生成债券信息矩阵'''
        #1 面值 2 年化利率
        itimes = len(self.schedule())-1
        indics = [[self.faceAmount]*itimes,[self.couponRate]*itimes] #浮息债需要重新计算
        mat = np.zeros(shape=(len(indics),itimes))
        for n in range(len(indics)):
            mat[n,:] = indics[n]
        return mat
    
    def date_process(self,sdate):
        '''计算给定日期前后的关键日期'''
        date_list = self.schedule()
        s_date = self.format_qldate(sdate)
        date_last = [x for x in date_list if x<s_date][-1] #上次付息日
        date_next = date_list[date_list.index(date_last)+1] #下次付息日
        unpay_date1 = [x for x in date_list if x>s_date] #剩余付息日 #+ql.Period(-1,ql.Years)
        unpay_date2 = date_list[date_list.index(unpay_date1[0])-self.frequency:] #添加最近一个付息日前推一年
        #计算给定日期所在的计息年份
        date_year = [x for x in date_list if x.year() ==s_date.year()]
        #考虑年终或者年初的假期对利息计算年份的选择的影响
        if s_date<date_year[0]:
            date_syear=[x for x in date_list if x.year() ==(s_date.year()-1)]
        elif s_date>date_year[-1]:
            date_syear=[x for x in date_list if x.year() ==(s_date.year()+1)]
        else:
            date_syear = date_year
        #考虑年终或者年初的假期对利息计算年份天数的付息区间选择
        if date_syear[-1] == date_list[-1]:
            year_days = date_list[-1]-date_list[-1-self.frequency]
        elif date_syear[0] == date_list[0]:
            year_days = date_list[self.frequency]-date_list[0]
        else:
            if len(date_syear)<self.frequency:
                year_days=date_list[date_list.index(date_syear[-1])+1]-\
                                   date_list[date_list.index(date_syear[0])-1]
            elif len(date_syear)>self.frequency:
                year_days=date_syear[-1]-date_syear[0]
            else:
                if date_syear[0].month()==1:
                    year_days=date_list[date_list.index(date_syear[-1])+1]-date_syear[0]
                else:
                    year_days=date_syear[-1]-date_list[date_list.index(date_syear[0])-1]
        return {'date_last':date_last,'date_next':date_next,'unpay_date1':unpay_date1,
                'unpay_date2':unpay_date2,'year_days':year_days}
        
    def accrued_interest(self,sdate):
        '''应计利息'''
        #我国债券主要用ACT/ACT（银行间市场，交易所市场的贴现债券），NL/365（交易所市场的非贴现债券）
        date_list = self.schedule()
        mat_i = self.info_matrix()
        s_date = self.format_qldate(sdate)
        date_last = self.date_process(sdate)['date_last']
        date_next = self.date_process(sdate)['date_next']
        location = date_list.index(date_next)-1
        if self.exchange == 'IB' or (self.exchange == 'EXC' and self.bondType == 5):
            fac = (s_date-date_last)/self.date_process(sdate)['year_days']           
            return mat_i[0,location]*mat_i[1,location]*fac 
        else:
            try:
                ql.Date(29,2,date_last.year())
            except RuntimeError:
                try:
                    ql.Date(29,2,date_next.year())
                except RuntimeError:
                    fac = (s_date-date_last)/365
                    return mat_i[0,location]*mat_i[1,location]*fac
                else:
                    if date_last<=ql.Date(29,2,date_next.year())<=date_next:
                        fac = (s_date-date_last-1)/365
                        return mat_i[0,location]*mat_i[1,location]*fac
                    else:
                        fac = (s_date-date_last)/365
                        return mat_i[0,location]*mat_i[1,location]*fac
            else:
                if date_last<=ql.Date(29,2,date_last.year())<=date_next:
                    fac = (s_date-date_last-1)/365
                    return mat_i[0,location]*mat_i[1,location]*fac
                else:
                    fac = (s_date-date_last)/365
                    return mat_i[0,location]*mat_i[1,location]*fac
            
    def FV(self,sdate):
        '''计算到期兑付日债券本息和'''
        s_date = self.format_qldate(sdate)
        pay_dates = self.schedule()
        info_mat = self.info_matrix()
        if s_date>pay_dates[-1] or s_date<pay_dates[0]:
            raise ValueError('date NotStarted or OutofDate')
        else:
            unpay_date2 = self.date_process(sdate)['unpay_date2']
#            lyear_days = unpay_date2[1]-unpay_date2[0] #付息期当年实际天数
#            remaining_days = unpay_date2[1]-s_date
            #1.	对于处于最后付息周期的固定利率债券、待偿期在一年及以内的到期一次还本付息债券和零息债券、贴现债
            #2.	对待偿期在一年以上的到期一次还本付息债券和零息债券，到期收益率按复利计算。
            ##非浮息债券
            if (unpay_date2[-2]<=s_date<unpay_date2[-1] and (self.bondType == 1 or self.bondType == 2))\
               or (self.bondType == 4 or self.bondType == 5):
                #付息周期等于一年的固定利率债券为M+C
                if self.frequency == 1 and (self.bondType == 1 or self.bondType == 2):
                    return info_mat[0,-1]*(1+info_mat[1,-1])
                #付息周期小于一年且按实际天数付息的固定利率债券
                elif self.frequency>1 and self.bondType == 1:
                    return info_mat[0,-1]+info_mat[0,-1]*info_mat[1,-1]*\
                         ((pay_dates[-1]-pay_dates[-2])/(pay_dates[-1]-unpay_date2[-1-self.frequency]))
                #付息周期小于一年且按平均值付息的固定利率债券
                elif self.frequency>1 and self.bondType == 2:
                    return info_mat[0,-1]+info_mat[0,-1]*info_mat[1,-1]/self.frequency
                #到期一次还本付息债券
                elif self.bondType == 4:
                    return info_mat[0,0]+sum(np.multiply(info_mat[0],info_mat[1]))
                #零息债券和贴现债
                elif self.bondType == 5:
                    return info_mat[0,0]
            else:
                raise ValueError('no need for calculating FV')
            #3.	对不处于最后付息周期的固定利率债券，到期收益率按复利计算。无需计算FV
            #4.	分次兑付债券.无需计算FV
            ##浮息债券
            
    def PV(self,sdate):
        '''债券全价'''
        s_date = self.format_qldate(sdate)
        pay_dates = self.schedule()
        info_mat = self.info_matrix()
        date_last = self.date_process(sdate)['date_last']
        date_next = self.date_process(sdate)['date_next']
        year_days = self.date_process(sdate)['year_days']
        fac = (s_date-date_last)/year_days #ACT/ACT
        return self.cleanPrice+self.faceAmount*info_mat[1, pay_dates.index(date_next)-1]*fac
        
    def YTM(self,sdate):
        '''到期收益率'''
        s_date = self.format_qldate(sdate)
        pay_dates = self.schedule()
        date_last = self.date_process(sdate)['date_last']
        date_next = self.date_process(sdate)['date_next']
        unpay_date1 = self.date_process(sdate)['unpay_date1']
        unpay_date2 = self.date_process(sdate)['unpay_date2']
        year_days = self.date_process(sdate)['year_days']
        if (unpay_date2[-2]<s_date<=unpay_date2[-1] and (self.bondType == 1 or self.bondType == 2))\
               or ((self.bondType == 4 or self.bondType == 5) and s_date>=unpay_date2[-1-self.frequency]):
            return ((self.FV(sdate)-self.PV(sdate))/self.PV(sdate))/((pay_dates[-1]-\
                   s_date)/year_days)
        elif (self.bondType == 4 or self.bondType == 5) and s_date<unpay_date2[-1-self.frequency]:
#            效率低下的方程求解
#            ytm_res=sy.Symbol('ytm_res')
#            year_num = math.floor((len(unpay_date1)-1)/self.frequency)
#            res = sy.solve(self.PV(sdate) - self.FV(sdate)/((1+ytm_res)**(
#                    (unpay_date1[-1-self.frequency*year_num]-s_date)/year_days+year_num)),ytm_res)
#            return res
            #效率高很多的非线性方程求解
            year_num = math.floor((len(unpay_date1)-1)/self.frequency)
            def f1(x):
                return self.PV(sdate) - self.FV(sdate)/((1+x)**(
                         (unpay_date1[-1-self.frequency*year_num]-s_date)/year_days+year_num))
            res = fsolve(f1,[1])
            #误差为：f1(res)
            return res[0]
        elif s_date<=unpay_date2[-2] and (self.bondType == 1 or self.bondType == 2):
            def f2(x):
                mid_equal = 0
                for n in range(len(unpay_date1)): 
                    mid_equal += (self.faceAmount*self.couponRate/self.frequency)/((
                            1+x/self.frequency)**((date_next-s_date)/(date_next-date_last)+n))
                return self.PV(sdate)-mid_equal-self.faceAmount/((1+x/self.frequency\
                                 )**((date_next-s_date)/(date_next-date_last)+len(unpay_date1)-1))
            res2 = fsolve(f2,[0])
            return res2[0]
        else:
            #其它类型债券（分次兑付债券，浮动利率债券）
            pass

    def duration(self,sdate):
        '''久期'''
        s_date = self.format_qldate(sdate)
        pay_dates = self.schedule()
        date_last = self.date_process(sdate)['date_last']
        date_next = self.date_process(sdate)['date_next']
        unpay_date1 = self.date_process(sdate)['unpay_date1']
        unpay_date2 = self.date_process(sdate)['unpay_date2']
#        year_days = self.date_process(sdate)['year_days']
        info_mat = self.info_matrix()
        if (unpay_date2[-2]<s_date<=unpay_date2[-1] and (self.bondType == 1 or self.bondType == 2))\
               or ((self.bondType == 4 or self.bondType == 5) and s_date>=unpay_date2[-1-self.frequency]):
            try:
                ql.Date(29,2,date_last.year())
            except RuntimeError:
                try:
                    ql.Date(29,2,date_next.year())
                except RuntimeError:
                    return (pay_dates[-1]-s_date)/365
                else:
                    if date_last<=ql.Date(29,2,date_next.year())<=date_next:
                        return (pay_dates[-1]-s_date)/366
                    else:
                        return (pay_dates[-1]-s_date)/365
            else:
                if date_last<=ql.Date(29,2,date_last.year())<=date_next:
                    return (pay_dates[-1]-s_date)/366
                else:
                    return (pay_dates[-1]-s_date)/365
        elif (s_date<=unpay_date2[-2] and (self.bondType == 1 or self.bondType == 2)) or\
               ((self.bondType == 4 or self.bondType == 5) and s_date<unpay_date2[-1-self.frequency]):
            Ci = info_mat[1,pay_dates.index(date_next)]
            f = self.frequency
            w = (date_next-s_date)/(date_next-date_last)
            n = len(unpay_date1)
            y = self.YTM(sdate)
            P = self.PV(sdate)
            F = info_mat[0,pay_dates.index(date_next)]
            mid_res = 0
            for i in range(n):
                mid_res += Ci/f*(w+i)/(1+y/f)**(w+i)
            return (mid_res+F*(w+n-1)/(1+y/f)**(w+n-1))/(f*P)
        else:
            pass

    def  modified_duration(self,sdate):
        '''修正久期'''
        D = self.duration(sdate)
        y = self.YTM(sdate)
        f = self.frequency
        return D/(1+y/f)

    def convexity(self,sdate):
        '''凸性'''
        s_date = self.format_qldate(sdate)
        info_mat = self.info_matrix()
        pay_dates = self.schedule()
        date_last = self.date_process(sdate)['date_last']
        date_next = self.date_process(sdate)['date_next']
        unpay_date1 = self.date_process(sdate)['unpay_date1']
        unpay_date2 = self.date_process(sdate)['unpay_date2']
        
        y = self.YTM(sdate)
        d = unpay_date1[-1]-s_date
        TY = self.date_process(sdate)['year_days']
        t = d/TY
        
        Ci = info_mat[1,pay_dates.index(date_next)]
        f = self.frequency
        w = (date_next-s_date)/(date_next-date_last)
        n = len(unpay_date1)
        P = self.PV(sdate)
        F = info_mat[0,pay_dates.index(date_next)]
        
        if (unpay_date2[-2]<s_date<=unpay_date2[-1] and (self.bondType == 1 or self.bondType == 2))\
               or ((self.bondType == 4 or self.bondType == 5) and s_date>=unpay_date2[-1-self.frequency]):
            return 2*(t**2)/((1+y*t)**2)
        elif (s_date<=unpay_date2[-2] and (self.bondType == 1 or self.bondType == 2)) or\
               ((self.bondType == 4 or self.bondType == 5) and s_date<unpay_date2[-1-self.frequency]):
            mid_res = 0
            for i in range(n):
                mid_res += (Ci/f)*(w+i)*(w+i+1)/(1+y/f)**(w+i)
            return (mid_res+F*(w+n-1)*(w+n)/(1+y/f)**(w+n-1))/((f**2)*P*((1+y/f)**2))
        else:
            pass
    
    def bvalue(self,sdate):
        '''基点价值'''
        return self.modified_duration(sdate)*(10**(-4))*self.PV(sdate)

#计算组合指标
class Portfolio_Indacates:    
    def __init__(self,bname,mvalue,cost,mduration,convexity,bvalue):
        self.bname = bname
        self.mvalue = mvalue
        self.cost = cost
        self.mduration = mduration
        self.convexity = convexity
        self.bvalue = bvalue

    @classmethod
    def matrix_info(cls,bond_list,bond_plus):
        mat_info = np.zeros(shape=(len(bond_list),5))
        for n in range(len(bond_list)):
            bond_cls_tmp = Bond_Profile(*eval(bond_list[n])['bond_info'])
            sdate_tmp = eval(bond_list[n])['date']
            mduration_tmp = bond_cls_tmp.modified_duration(sdate_tmp)
            convexity_tmp = bond_cls_tmp.convexity(sdate_tmp)
            bvalue_tmp = bond_cls_tmp.bvalue(sdate_tmp)
            mat_info[n,:] = [bond_plus['market_value'][n],bond_plus['cost'][n],
                                         mduration_tmp,convexity_tmp,bvalue_tmp]
        res = cls((eval(bond_list[0])['bond_info'][0],eval(bond_list[1])['bond_info'][0]),*mat_info.T)
        return res

    def p_duration(self,style):
        '''组合久期'''
        if style == 'marketvalue':
            return np.average(self.mduration,weights=self.mvalue)
        elif style == 'cost':
            return np.average(self.mduration,weights=self.cost)
        else:
            raise ValueError('wrong style')
    
    def p_convexity(self,style):
        '''组合凸性'''
        if style == 'marketvalue':
            return np.average(self.convexity,weights=self.mvalue)
        elif style == 'cost':
            return np.average(self.convexity,weights=self.cost)
        else:
            raise ValueError('wrong style')

    def p_bvalue(self,style):
        '''组合基点价值'''
        if style == 'marketvalue':
            return np.average(self.bvalue,weights=self.mvalue)
        elif style == 'cost':
            return np.average(self.bvalue,weights=self.cost)
        else:
            raise ValueError('wrong style')

class indicates_result:        
    @staticmethod
    def dict_round(dic,n=4):
        '''结果保留四位小数'''
        for key in dic.keys():
            try:
                dic[key] = round(dic[key],n)
            except:
                continue
        return dic
    
    @classmethod
    def single_bond(cls,bond):
        sdate = bond['date']
        bond_cls = Bond_Profile(*bond['bond_info'])
        bond_res = {'1-债券代码':bond_cls.code,
                    '2-应计利息':bond_cls.accrued_interest(sdate),
                    '3-债券全价':bond_cls.PV(sdate),
                    '4-到期收益率':bond_cls.YTM(sdate),
                    '5-久期':bond_cls.duration(sdate),
                    '6-修正久期':bond_cls.modified_duration(sdate),
                    '7-凸性':bond_cls.convexity(sdate),
                    '8-基点价值':bond_cls.bvalue(sdate),
                    }
    
        return cls.dict_round(bond_res)
    
    @classmethod
    def port_bonds(cls,bond_list,bond_plus):
        bonds_cls = Portfolio_Indacates.matrix_info(bond_list,bond_plus)
        bonds_res = {'1-债券代码':bonds_cls.bname,
                     '2-修正久期(市值加权)':bonds_cls.p_duration('marketvalue'),
                     '3-修正久期(成本加权)':bonds_cls.p_duration('cost'),
                     '4-凸性(市值加权)':bonds_cls.p_convexity('marketvalue'),
                     '5-凸性(成本加权)':bonds_cls.p_convexity('cost'),
                     '6-基点价值(市值加权)':bonds_cls.p_bvalue('marketvalue'),
                     '7-基点价值(成本加权)':bonds_cls.p_bvalue('cost'),
                     }
        return cls.dict_round(bonds_res)

if __name__=='__main__':
    #单个债券
    bond1={'bond_info':('180210.IB',100,103.5811,'2018-07-06','2028-07-06',0.0404,1,1,0,'IB','zhongzhai'),
       'date':'2019-01-11'}
    bond2={'bond_info':('user_defined',100,98.5,'2010-01-01','2020-01-01',0.05,2,1,0,'IB','zhongzhai'),
           'date':'2019-01-11'}
    
    print (indicates_result.single_bond(bond1))
    indicates_result.single_bond(bond2)
    
    #组合债券
    bond_list = ['bond1','bond2']
    bond_plus = {'market_value':[500,380],'cost':[0.4,0.35]}
    
    indicates_result.port_bonds(bond_list,bond_plus)

#self =Bond_Profile('180210.IB',100,103.5811,'2018-07-06','2028-07-06',0.0404,1,1,0,'IB','zhongzhai')