# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 23:47:04 2019

@author: wuwangchuxin
"""

#付息日矩阵 面值矩阵  利率矩阵 生成未来现金流 （利息加本金）
#未来

#fixed income indicators calculating
import numpy as np
#import datetime 
import QuantLib as ql
#import sympy as sy
#import math
from scipy.optimize import fsolve
import pandas as pd

# 输入
# 债券名称、起息日、票面利率，票面利率，当前余额
# 债券基本信息

#self = Bond_Profile('180210.IB','2018-07-03',100,'IB','2015-02-28','2025-02-28',2,100,1,
#                        np.nan,np.nan,1,0.0404,np.nan,np.nan,1,3,'2018-03-12',103.20,0,0)
class Bond_Profile:
    def __init__(self,Code,IssueDate,IssuingPrice,Exchange,ValueDate,Maturity,Frequency,FaceAmount,
                 PrincipalPaymentType,PrincipalPayments,DuePayment,BondType,CouponRate,CouponRateStandard,
                 RateMargin,DaysCal,InterestType,CalDate,CleanPrice,IsOption,IsABS,**kw):
        '''债券基本信息，类似PrincipalPaymentType=3时，还本是不规则的情况，PrincipalPayments需要在实例化
           时输入各期未偿还面值向量，PrincipalPaymentType取1或2时，PrincipalPayments设定为空值，比如np.nan；
           同样的情况还适用于CouponRate，CouponRateStandard，RateMargin属性'''       
        self.Code = Code #债券代码
        self.IssueDate = IssueDate #发行日期
        self.IssuingPrice = IssuingPrice #发行价格
        self.Exchange = Exchange #交易市场：IB银行间 SSE交易所

        self.ValueDate = ValueDate #计息日期
        self.Maturity = Maturity #到期日期
        self.Frequency = Frequency #年付息频率(1 Annual 2 Semiannual 4 Quarterly 12 Monthly)
        
        self.FaceAmount = FaceAmount #面值
        self.PrincipalPaymentType = PrincipalPaymentType #还本方式：1到期一次还本；2均匀分期还本；3其它分期还本
        self.PrincipalPayments = PrincipalPayments #其它分期还本矩阵(上一字段取值为3,本字段才有取值)
        self.DuePayment = DuePayment #贴现债券到期兑付额
#        if self.PrincipalPaymentType == 3:
#           self.PrincipalPayments = eval(input('please input matrix of principal payments:'))
        
        #1固定利息;2浮动利息;3到期一次还本付息(国内一般为单利即利随本清，暂不考虑复利）;
        #4零息债券（以投资额即本金为基准计算利率）、贴现债（以偿还额即本息和为基准计算利率）
        self.BondType = BondType
        self.CouponRate = CouponRate #固定利息、到期一次还本付息和利随本清债券利率
        self.CouponRateStandard = CouponRateStandard #浮动利息债利率基准
        self.RateMargin = RateMargin #浮息债利差
#        if self.BondType in [1,3]:
#           self.CouponRate = eval(input('please input coupon rate:')) #1和3类型的债券票面利率
#        if self.BondType == 2:
#           self.CouponRateStandard = eval(input('please input coupon rate standard:')) #浮息债基准
#           self.RateMargin = eval(input('please input coupon rate margin:'))#浮息债利差
        self.DaysCal = DaysCal #1按实际天数付息;2按平均值付息
        self.InterestType = InterestType #利率类型（1单利，2按年复利；3按付息期复利；4连续复利；5其它）
        
        self.CalDate = CalDate #所要计算的日期
        self.CleanPrice = CleanPrice #净价
#        self.DirtyPrice = DirtyPrice #全价
        
        self.IsOption = IsOption #是否含权债
        self.IsABS = IsABS #是否资产支持证券
#        self.Standard = Standard #计算基准：默认中债
#        self.NowDate = datetime.datetime.now().strftime('%Y-%m-%d') #d当前日期
        #补充自定义信息
        for info in kw:
            setattr(self,info,kw[info])
    
    @staticmethod
    def format_qldate(datestr):
        '''将字符串日期转换为quantlib类型的日期'''
        return ql.Date(*[int(x) for x in datestr.split('-')[::-1]])
    
    def schedule(self):
        '''返回付息日期序列
           国内债券和存款计息天数调整规则默认为不调整
           考虑月末法则'''
        sdate = self.format_qldate(self.ValueDate)
        edate = self.format_qldate(self.Maturity)
        freq = ql.Period(self.Frequency) #ql.Annual
        calendar = ql.China()
        schedule = ql.Schedule(sdate, edate,freq,calendar,ql.Unadjusted,
                                ql.Unadjusted,ql.DateGeneration.Forward, 
                              ql.Date.isEndOfMonth(self.format_qldate(self.ValueDate)))        
#        schedule = ql.Schedule(sdate, edate, freq, calendar, ql.Unadjusted,ql.Unadjusted, 
#                                   ql.DateGeneration.Forward, False)
#        schedule = ql.Schedule(sdate, edate, freq, calendar, ql.Following,ql.Following, 
#                                   ql.DateGeneration.Backward, False)
        return list(schedule)
   
    def date_process(self):
        '''计算给定日期前后的关键日期；
           规则：ACT/ACT,给定日期距付息日实际天数除以给定日期所在债券自身付息年度的实际天数得到年化日期,
           上述规则根据文档前述方法计算（比如应计利息表格变量含义的TY计算方法）,不同于文档附录4.2.5中关于
           ACT/ACT的描述：属于同一计息期的闰年部分按照闰年天数计算加上属于非闰年的按非闰年的天数计算的部分'''
        date_list = self.schedule()
        caldate = self.format_qldate(self.CalDate)  #给定日期
        date_last = [x for x in date_list if x<caldate][-1] #给定日期上次付息日
        date_next = date_list[date_list.index(date_last)+1] #给定日期下次付息日
        unpay_date1 = [x for x in date_list if x>caldate] #剩余付息日 #+ql.Period(-1,ql.Years)
        unpay_date2 = date_list[date_list.index(unpay_date1[0])-self.Frequency:] #添加最近一个付息日前推一年
        year_ord = sorted(list(range(1,int((len(date_list)-1)/self.Frequency+1)))*self.Frequency) #计息年度序列
        year_days = []  #所在付息年度的天数
        for ny in range(1,max(year_ord)+1):
            mid_res = date_list[ny*self.Frequency]-date_list[(ny-1)*self.Frequency]
            year_days+=[mid_res]*self.Frequency
        period_days = np.array(date_list[1:])-np.array(date_list[:-1]) #所在付息区间天数        
        year_interval = [x for x in period_days/year_days] #付息日之间年化时间
        ord_tmp = date_list[1:].index(date_next)
        next_interval = (date_next - caldate)/year_days[ord_tmp] #caldate距下次付息日年化时间
        last_interval = (caldate - date_last)/year_days[ord_tmp] #caldate距上次付息日年化时间
        #计算给定日期所在的计息年份
        caldate_year_ord = year_ord[date_list[1:].index(date_next)] #所在计息年份序数
        caldate_year_days = year_days[date_list[1:].index(date_next)] #所在计息年份天数
        
        #指定日期距离各付息日的年化时间
        index_nextpd = date_list[1:].index(date_next)
        caldate_interval = [0]*(len(date_list)-1)
        #向后的时间
        for cdy in range(index_nextpd,len(date_list)-1):
                caldate_interval[cdy] = next_interval+sum(year_interval[index_nextpd+1:cdy+1])
        #向前的时间
        for cdy2 in list(range(index_nextpd))[::-1]:
                caldate_interval[cdy2] = -(last_interval+sum(year_interval[cdy2+1:index_nextpd]))
                
        return {'date_last':date_last,'date_next':date_next,'unpay_date1':unpay_date1,
                'unpay_date2':unpay_date2,'year_ord':year_ord,'year_days':year_days,
                'period_days':list(period_days),'caldate_year_ord':caldate_year_ord,
                'caldate_year_days':caldate_year_days,'year_interval':year_interval,
                'next_interval':next_interval,'last_interval':last_interval,
                'caldate_interval':caldate_interval}

    def info_df(self):
        '''计算基础指标，生成债券信息dataframe'''
        #1付息日;2面值;3年化利率;4未来现金流;
        date_list = self.schedule()
        itimes = len(date_list)-1
        schedule_vec = date_list[-itimes:] #所有付息日序列
        caldate_dic = self.date_process()
        year_interval_vec = caldate_dic['year_interval'] #付息日间年化时间间隔
        caldate_interval_vec = caldate_dic['caldate_interval'] #指定日期距离各付息日年化时间
            
        #剩余面值为付息日截止前，因为当日偿还本金后下一付息周期才不用偿还这部分本金产生的利息
        if self.PrincipalPaymentType == 1:
            FaceAmount_vec = [*[self.FaceAmount]*(itimes-1),0]
        elif self.PrincipalPaymentType == 2:
            every_amount = self.FaceAmount/itimes
            mid_vec = []
            for ni in range(1,itimes+1):
                mid_vec.append(self.FaceAmount-every_amount*ni)
            FaceAmount_vec = mid_vec
        else:
            FaceAmount_vec = self.PrincipalPayments
        #利率向量
        if self.BondType in [1,3]:
            CouponRate_vec = [self.CouponRate]*itimes
        elif self.BondType == 2:
            CouponRate_vec = np.array(self.CouponRateStandard) + np.array(self.RateMargin)
        else:
            CouponRate_vec = [0]*itimes
        FaceAmount_vec_adj = np.array([self.FaceAmount,*FaceAmount_vec[:-1]])
        InterestFlow_vec=list(np.array(CouponRate_vec)*FaceAmount_vec_adj*np.array(year_interval_vec))
        FaceAmountFlow_vec = list(FaceAmount_vec_adj-np.array(FaceAmount_vec))
        CashFlow_vec = list(np.array(FaceAmountFlow_vec) + np.array(InterestFlow_vec))
        return pd.DataFrame([schedule_vec,year_interval_vec,caldate_interval_vec,caldate_dic['year_ord'],
                             caldate_dic['year_days'],caldate_dic['period_days'],FaceAmount_vec,
                             CouponRate_vec,InterestFlow_vec,FaceAmountFlow_vec,CashFlow_vec],
                             index=['DateList','YearInterval','CaldateInterval_back','YearOrd','YearDays',
                                    'PeriodDays','FaceAmount_back','CouponRate','InterestFlow',
                                    'FaceAmountFlow','CashFlows'],
                             columns=range(1,itimes+1))
        
    def accrued_interest(self):
        '''应计利息'''
        #我国债券主要用ACT/ACT（银行间市场，交易所市场的贴现债券），NL/365（交易所市场的非贴现债券）        
        date_list = self.schedule()[1:]
        df_i = self.info_df() #test = df_i.iloc[1:,:]
        caldate = self.format_qldate(self.CalDate)
        date_next = self.date_process()['date_next']

        col_name_caldate = date_list.index(date_next)+1
        if self.Exchange == 'IB':
            if self.BondType in (1,2) or self.IsABS==1:
#                C_ind = df_i.loc['CouponRate',col_name_caldate]*100 #每百元面值年利息
#                t_ind = caldate-self.date_process()['date_last']#起息日或上一付息日至估值日的实际天数
#                TY_ind = df_i.loc['YearDays',col_name_caldate]#本付息周期所在计息年度的实际天数
#                TS_ind = df_i.loc['PeriodDays',col_name_caldate]#本付息周期的实际天数
#                if self.Frequency == 1:
#                    return (C_ind/TS_ind)*t_ind*(m_ind/100)
#                elif self.Frequency>1 and self.DaysCal==1:
#                    return (C_ind/TY_ind)*t_ind*(m_ind/100)
#                elif self.Frequency>1 and self.DaysCal==2:
#                    return (C_ind/self.Frequency)*(t_ind/TS_ind)*(m_ind/100)
                if col_name_caldate==1:
                    m_ind = self.FaceAmount
                else:
                    m_ind = df_i.loc['FaceAmount_back',col_name_caldate-1]#百元面值当前剩余本金值
                if self.Frequency == 1 or (self.Frequency>1 and self.DaysCal==1):
                    return df_i.loc['CouponRate',col_name_caldate]*(caldate-\
                            self.date_process()['date_last'])/df_i.loc['YearDays',col_name_caldate]*m_ind
                elif self.Frequency>1 and self.DaysCal==2:
                    return df_i.loc['CouponRate',col_name_caldate]/self.Frequency*(caldate-\
                            self.date_process()['date_last'])/df_i.loc['PeriodDays',col_name_caldate]*m_ind
            elif self.BondType == 3:
                mid_y = range(1,col_name_caldate)
                return sum(df_i.loc['CouponRate',mid_y]*df_i.loc['YearInterval',mid_y]*100)+\
                         (-df_i.loc['CouponRate',col_name_caldate]*
                           df_i.loc['CaldateInterval',col_name_caldate-1]*100)
            elif self.BondType == 4:
                M_ind = self.DuePayment
                pd_ind = self.IssuingPrice
                T_ind = self.schedule()[-1]-self.schedule()[0]
                t_ind = caldate-self.schedule()[0]
                return (M_ind-pd_ind)*t_ind/T_ind
        elif self.Exchange == 'SSE':
            if self.BondType in (1,2,3) or self.IsABS==1:
                C_ind = df_i.loc['CouponRate',col_name_caldate]*100
                if self.BondType==3:
                    t_ind = caldate-self.schedule()[0]
                else:
                    t_ind = caldate-self.date_process()['date_last']
                if col_name_caldate==1:
                    m_ind = self.FaceAmount
                else:
                    m_ind = df_i.loc['FaceAmount_back',col_name_caldate-1]#百元面值当前剩余本金值
                return (C_ind/365)*t_ind*(m_ind/100)
            elif self.BondType==4:
                M_ind = self.DuePayment
                pd_ind = self.IssuingPrice
                T_ind = self.schedule()[-1]-self.schedule()[0]
                t_ind = caldate-self.schedule()[0]
                return (M_ind-pd_ind)*t_ind/T_ind
            
#    def FV(self):
#        '''计算到期兑付日债券本息和'''
#        s_date = self.format_qldate(self.CalDate)
#        pay_dates = self.schedule()
#        df_i = self.info_df()
#        if s_date>pay_dates[-1] or s_date<pay_dates[0]:
#            raise ValueError('date NotStarted or OutofDate')
#        else:
#            unpay_date2 = self.date_process()['unpay_date2']
##            lyear_days = unpay_date2[1]-unpay_date2[0] #付息期当年实际天数
##            remaining_days = unpay_date2[1]-s_date
#            #1.	对于处于最后付息周期的固定利率债券、待偿期在一年及以内的到期一次还本付息债券和零息债券、贴现债
#            #2.	对待偿期在一年以上的到期一次还本付息债券和零息债券，到期收益率按复利计算。
#            ##非浮息债券
#            if (unpay_date2[-2]<=s_date<unpay_date2[-1] and self.BondType == 1)\
#               or (self.BondType == 3 or self.BondType == 4):
#                #付息周期等于一年的固定利率债券为M+C
#                if self.frequency == 1 and self.BondType == 1:
#                    return df_i.iloc[0,-1]*(1+info_mat[1,-1])
#                #付息周期小于一年且按实际天数付息的固定利率债券
#                elif self.frequency>1 and self.bondType == 1:
#                    return info_mat[0,-1]+info_mat[0,-1]*info_mat[1,-1]*\
#                         ((pay_dates[-1]-pay_dates[-2])/(pay_dates[-1]-unpay_date2[-1-self.frequency]))
#                #付息周期小于一年且按平均值付息的固定利率债券
#                elif self.frequency>1 and self.bondType == 2:
#                    return info_mat[0,-1]+info_mat[0,-1]*info_mat[1,-1]/self.frequency
#                #到期一次还本付息债券
#                elif self.bondType == 4:
#                    return info_mat[0,0]+sum(np.multiply(info_mat[0],info_mat[1]))
#                #零息债券和贴现债
#                elif self.bondType == 5:
#                    return info_mat[0,0]
#            else:
#                raise ValueError('no need for calculating FV')
#            #3.	对不处于最后付息周期的固定利率债券，到期收益率按复利计算。无需计算FV
#            #4.	分次兑付债券.无需计算FV
#            ##浮息债券
    def FV(self):
        '''计算到期兑付日债券本息和'''
        return self.info_df().iloc[-1,-1]
    
    def PV(self):
        '''债券全价'''
        return self.CleanPrice+self.accrued_interest()
#        caldate = self.format_qldate(self.CalDate)
#        date_pro = self.date_process()
#        pay_dates = self.schedule()
#        df_i = self.info_df()
#        date_last = date_pro['date_last']
#        date_next = date_pro['date_next']
#        year_days = date_pro['year_days']
#        fac = (caldate-date_last)/year_days #ACT/ACT
#        return self.cleanPrice+self.faceAmount*info_mat[1, pay_dates.index(date_next)-1]*fac
        
    def YTM(self):
        '''到期收益率'''
        caldate = self.format_qldate(self.CalDate)
        pay_dates = self.schedule()
        date_pro = self.date_process()
        date_last = date_pro['date_last']
        date_next = date_pro['date_next']
        unpay_date1 = date_pro['unpay_date1']
        df_i = self.info_df()
        
        mid_index = pay_dates[1:].index(date_next)
        
        if (df_i.iloc[0,-2]<caldate<=df_i.iloc[0,-1] and (self.BondType == 1 or self.BondType == 2)\
               or ((self.BondType == 3 or self.BondType == 4) and
                          df_i.iloc[0,-1-self.Frequency]<=caldate<df_i.iloc[0,-1]):
            return ((self.FV()-self.PV())/self.PV())/((pay_dates[-1]-caldate)/df_i.iloc[4,-1])               
        elif (self.BondType == 3 or self.BondType == 4) and caldate<df_i.iloc[0,-1-self.Frequency]:
#            效率低下的方程求解
#            ytm_res=sy.Symbol('ytm_res')
#            year_num = math.floor((len(unpay_date1)-1)/self.frequency)
#            res = sy.solve(self.PV(sdate) - self.FV(sdate)/((1+ytm_res)**(
#                    (unpay_date1[-1-self.frequency*year_num]-s_date)/year_days+year_num)),ytm_res)
#            return res
            #效率高很多的非线性方程求解
            year_num = df_i.iloc[3,-1]-df_i.iloc[3,mid_index]
            def f1(x):
                return self.PV()-self.FV()/((1+x)**((date_next-caldate)/df_i.iloc[4,mid_index]+year_num))
            res = fsolve(f1,[0.03])
            #误差为：f1(res)
            return res[0]
        elif caldate<=df_i.iloc[0,-2] and (self.BondType == 1 or self.BondType == 2):
            #用向量的方法没有得出解
#            def f2(x2):
#                C_div_f = np.array(list(df_i.loc['CashFlows',mid_index+1:]))
#                one_plus_y_div_f = np.array([1+x2/self.Frequency]*len(unpay_date1))
#                d_arr = np.array([date_next-caldate]*len(unpay_date1))
#                TS1 = np.array([df_i.loc['PeriodDays',mid_index+1]]*len(unpay_date1))
#                n_arr = np.array(range(len(unpay_date1)))
#                mid_res = C_div_f/(one_plus_y_div_f**(d_arr/TS1+n_arr))
#                return self.PV()-sum(mid_res)            
#            res2 = fsolve(f2,[0.03])
#            def f2(x2):
#                mid_equal = 0
#                for n in range(len(unpay_date1)):                     
#                    mid_equal += (self.FaceAmount*self.CouponRate/self.Frequency)/((
#                            1+x2/self.Frequency)**((date_next-caldate)/(date_next-date_last)+n))
#                return self.PV()-mid_equal-self.FaceAmount/((1+x2/self.Frequency\
#                                 )**((date_next-caldate)/(date_next-date_last)+len(unpay_date1)-1))
#            res2 = fsolve(f2,[0])
#            return res2[0]
            def f2(x2):
                mid_equal = 0
                for n in range(len(unpay_date1)):
                    C_div_f = df_i.loc['FaceAmount_back',mid_index+n]*\
                                df_i.loc['CouponRate',mid_index+1+n]/self.Frequency
                    d_div_TS1 = (date_next-caldate)/(date_next-date_last)                
                    mid_equal += C_div_f/((1+x2/self.Frequency)**(d_div_TS1+n))
                    
                return self.PV()-mid_equal-df_i.iloc[6,-2]/((1+x2/self.Frequency\
                                   )**(d_div_TS1+len(unpay_date1)-1))
            res2 = fsolve(f2,[0])
            return res2[0]
        else:
            #其它类型债券（分次兑付债券）
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
    print (indicates_result.single_bond(bond2))
    
    #组合债券
    bond_list = ['bond1','bond2']
    bond_plus = {'market_value':[500,380],'cost':[0.4,0.35]}
    
    print (indicates_result.port_bonds(bond_list,bond_plus))
    



