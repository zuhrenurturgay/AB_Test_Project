
########################
# AB TEST PROJECT
########################

########################
# İŞ PROBLEMİ
########################

# Facebook kısa süre önce mevcut maximum bidding adı verilen teklif
# verme türüne alternatif olarak yeni bir teklif türü olan average bidding’i tanıttı.
# Müşterilerimizden biri olan bombabomba.com, bu yeni özelliği test
# etmeye karar verdi ve averagebidding’in, maximumbidding’den daha
# fazla dönüşüm getirip getirmediğini anlamak için bir A/B testi yapmak istiyor.

########################
# VERİ SETİ HİKAYESİ
########################

# bombabomba.com’un web site bilgilerini içeren bu veri setinde kullanıcıların
# gördükleri ve tıkladıkları reklam sayıları gibi bilgilerin yanı sıra buradan gelen
# kazanç bilgileri yer almaktadır.
# Kontrol ve Test grubu olmak üzere iki ayrı veri seti vardır.

########################
# DEĞİŞKENLER
########################

# Impression – Reklam görüntüleme sayısı
# Click – Tıklama. Görüntülenen reklama tıklanma sayısını belirtir.
# Purchase – Satın alım. Tıklanan reklamlar sonrası satın alınan ürün sayısını belirtir.
# Earning – Kazanç. Satın alınan ürünler sonrası elde edilen kazanç

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, pearsonr, spearmanr, kendalltau, \
    f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

control_df= pd.read_excel("datasets/ab_testing.xlsx", sheet_name="Control Group")
test_df= pd.read_excel("datasets/ab_testing.xlsx", sheet_name="Test Group")

# Unnamed olarak görülen columnları sildik.
control_df=control_df.loc[:, ~control_df.columns.str.contains('Unnamed')].head()
test_df=test_df.loc[:, ~test_df.columns.str.contains('Unnamed')].head()

control_df.isnull().sum()
test_df.isnull().sum()
control_df.shape
test_df.shape

control_df["Purchase"].mean()
test_df["Purchase"].mean()

# Aykırı gözlem tespiti
sns.boxplot(control_df["Purchase"])
sns.boxplot(test_df["Purchase"])
# Aykırı gözlem yoktur

##########
# GÖREV-1:A/B testinin hipotezini tanımlayınız.
##########

# Control_df ve Test_df ortalamaları arasında karşılaştırma yapmak için A/B testini uygulayacağız.

# 1) Varsayım Kontrolü
# -Normallik Varsayımı

# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1: Normal dağılım varsayımı sağlanmamaktadır.
# p-value < 0.05 ise H0 reddedilir değilse kabul edilir ve normal dağılım varsayımı sağlanmaktadır.
# Normallik sağlanıyorsa bağımsız iki örneklem T testi yapılır.
# Normallik sağlanmıyorsa mannwithneyu testi yapılır

test_stat, pvalue = shapiro([control_df["Purchase"]])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

test_stat, pvalue = shapiro([test_df["Purchase"]])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# Varyans Homojenliği

# H0: Varyanslar Homojendir
# H1: Varyanslar Homojen Değildir

test_stat, pvalue = levene(test_df["Purchase"],control_df["Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p-value değeri 0.05 ten büyük oldupu için H0 reddedemeyiz. Varyanslar homojendir.
# control_df p-value değeri= 0.6576. p-value < 0.05 ten büyüktür, H0 reddedemeyiz ve normal dağılım
# varsayımı sağlanmaktadır.


# 2)Hipotezin Uygulanması
# Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
# Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)
# Varsayımlar ssağlandığına göre bağımsız iki örneklem T testi yapacağız.

test_stat, pvalue = ttest_ind(control_df["Purchase"], test_df["Purchase"],
                              equal_var=True)

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p-value < ise 0.05 'ten H0 RED.
# p-value < 0.05 değilse H0 REDDEDILEMEZ.
# Bağımsız iki örneklem T testinde p-value değeri 0.1685 tir, 0.05'ten büyük olduğu için H0 reddedilemez.

#########
# GÖREV-2: Çıkan test sonuçlarının istatistiksel olarak
# anlamlı olup olmadığını yorumlayınız.
# GÖREV-3: Hangi testleri kullandınız?
# Sebeplerini belirtiniz.
#########

# Bir önceki adımda AB testini hipotezini tanımladık, ilk adım Varsayım Kontrolü
# p-value < 0.05 ise H0 reddedilir.
# 1) Varsayım Kontrolü:
# - Normallik Varsayımı:
# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1: Normal dağılım varsayımı sağlanmamaktadır.
# p-value < 0.05 ise H0 reddedilir değilse kabul edilir ve normal dağılım varsayımı sağlanmaktadır.
# SONUÇ: p-value değeri=0.6576 çıktı. H0 reddedilemez ve normal dağılım varsayımı sağlanmaktadır.
# İki grubun ortalaması arasında istatistiki olarak anlamlı bir farklılık yoktur.
# - Varyans Homojenliği
# H0: Varyanslar Homojendir
# H1: Varyanslar Homojen Değildir.
# SONUÇ: p-value = 0.3659 çıkmıştır. p-value değeri 0.05!ten büyük çıktığı için H0 değerini reddedemeyiz.
# Varsayım homojendir.

#### Her iki p-value değeri de 0.05'ten büyük çıktığı için Varsayım Kontrolünde varsayımlar sağlanmıştır.
# Varsayım Kontrolünden sonraki adımımız Hipotezin Uygulanmasıdır.
# Varsayımlar sağlandığı için Bağımsız İki Örneklem T Testi yaparız. Eğer varsayımlar sağlanmasaydı mannwithneyu
# testi(nonparametrik test) yapardık lakin varsayımlar sağlandığı için Bağımsız İki Örneklem T testi yapıyoruz.

# 2) Hipotezin Uygulanması:
# - Varsayımlar sağlanıyorsa bağımsız iki örneklem T testi (parametrik test)
# SONUÇ: p-value= 0.1685, 0.05'ten büyüktür. Bu durumda H0 reddedilmez ve iki grup ortalamaları
# arasında istatistiki olarak anlamlı bir farklılık yoktur.

##########
# GÖREV-4: Görev 2’de verdiğiniz cevaba göre, müşteriye
# tavsiyeniz nedir?
##########

# Verileri istatistiki olarak test ettik ve sonuç olarak mevcut olan maximum bidding'e alternatik olarak tanıtılan
# average bidding'in verileriyle karşılaştırıldığında ortalamaları arasında şu anda anlamlı bir fark yoktur.
# Zaman geçtikçe örneklem sayısı artar, örneklem sayısı arttıkça ana kitlenin değerine erişiriz ve daha iyi örüntü
# elde edebiliriz.
# Müşterilerimizden biri olan bombabomba.com'a bu modeli bir süre daha test etmemizin uygun olacağını tavsiye ediyorum.













