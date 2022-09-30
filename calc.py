import math
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


# sample = [
# 0.76,0.82,0.70,0.86,0.78,0.96,0.68,0.83,0.92,0.86,0.86,0.84,0.66,0.92,0.76,0.95,0.84,0.91,0.78,0.70,0.78,0.70,0.82,0.99,0.83,0.86,0.67,0.91,0.75,0.86,0.83,0.75,0.95,0.79,0.65,0.84,0.78,0.88,0.70,0.95,0.87,0.71,0.92,1.00,0.75,0.87,0.80,0.79,0.66,0.90,0.79,0.82,0.65,0.83,0.88,0.96,0.75,0.91,0.71,0.87,0.76,0.90,0.71,0.87,0.74,0.94,0.80,1.00,0.95,0.79,0.96,0.98,0.84,0.79,0.91,0.71,0.65,0.90,0.88,0.74,0.74,0.67,0.94,0.72,1.01,0.82,0.80,0.83,0.99,0.83,0.88,0.80,0.72,0.91,0.84,0.74,0.94,0.72,0.83,0.87
# ]
sample = [
169	,
145	,
146	,
151	,
163	,
168	,
179	,
163	,
145	,
168	,
154	,
168	,
152	,
157	,
169	,
157	,
177	,
169	,
153	,
167	,
143	,
122	,
142	,
178	,
165	,
143	,
162	,
155	,
162	,
141	,
155	,
163	,
132	,
149	,
148	,
179	,
149	,
152	,
142	,
148	,
113	,
117	,
152	,
195	,
151	,
165	,
146	,
175	,
173	,
152	,
155	,
165	,
161	,
145	,
153	,
159	,
113	,
177	,
174	,
158	,
171	,
132	,
148	,
166	,
139	,
149	,
151	,
131	,
168	,
152	,
168	,
159	,
136	,
182	,
166	,
141	,
152	,
154	,
153	,
155	,
153	,
107	,
138	,
135	,
138	,
102	,
143	,
174	,
185	,
184	,
136	,
125	,
149	,
136	,
128	,
169	,
157	,
182	,
168	,
181	
]


maximum = max(sample)
minimum = min(sample)
least = minimum
range_sample = abs(maximum - minimum)
print ("Максимальное значение =", maximum, "\nМинимальное значение =",minimum, "\nРазмах выборки =", range_sample)
size = len(sample)
amount_of_interval = 1 + math.floor(3.322*math.log(size,10))
#amount_of_interval =9 
print("Объем выборки =", size, "\nКоличество интервалов =", amount_of_interval)
length = round((range_sample/amount_of_interval),5)

print("Длина одного интервала =",length)
r_a_o_i = amount_of_interval
dF = 0
Q = 0
Y = 0
Z = 0
V = 0

X = np.mean(sample)
D = np.var(sample)
D_ = np.var(sample)*(size/(size-1))
sigma = math.sqrt(np.var(sample))
s = math.sqrt(np.var(sample))*math.sqrt(size/(size-1))
print("Математическое ожидание M(X) =", X)
print("Дисперсия D(X) =", D)
print("Уточненная дисперсия s²(X) =", D_)
print("Среднее квадратичное отклонение σ(X) =", sigma)
print("Уточненное среднее квадратичное отклонение s(X) =",s)
array_y = []
array_x = []

n_ = 0
for t in range(0, int(r_a_o_i)):
    start = minimum
    finish = None 
    if t == r_a_o_i-1:
        finish = maximum
    else:
        finish = minimum+length
    print("Начало интервала =", round((start),5),"\nКонец интервала =", round((finish),5))
    #for i in range(size):
     #   print(([1 for k in sample if sample[i]>=start and sample[i]<=finish]))
    i = None
    centre = (start+finish)/2
    print ("Центр интервала =",centre)
    minimum = finish
    amount = len(list(filter(lambda x: x>=start and x<=finish, sample)))
    print ("Количество, эмпирически входящих в данный интервал =",amount)
    z_i_1 = (start - np.mean(sample))/math.sqrt(np.var(sample))
    z_i_2 = (finish - np.mean(sample))/math.sqrt(np.var(sample))
    if start == least:
        z_i_1 = -5
    if finish == maximum:
        z_i_2 = 5
        
    print ("z_i1 =",z_i_1,"\nz_i2 =",z_i_2)
    F1 = (math.erf(z_i_1/math.sqrt(2)))/2
    F2 = (math.erf(z_i_2/math.sqrt(2)))/2
    if z_i_1 == -5:
        F1 = -0.5
    if z_i_2 == 5:
        F2 = 0.5
    
    print ("F1 =", F1, "\nF2 =", F2)
    p = F2 - F1
    print ("Δp = F2-F1 =", p)
    dF = dF + p
    n_ = round(p*size)
    print ("Количество, теоретически входящих в данный интервал =",n_)
    err = ((amount - n_)**2)/n_
    print ('Ошибка = ', err)
    Q = Q + n_
    Y = Y + err
    nnn = ((amount-n_)**2)/n_
    Z = Z + nnn
    array_y.append(amount/size)
    array_x.append(centre)

    nn = (amount**2)/n_
    
    V = V + nn
    print("========================================")
    

print ('χ² наблюдаемое =', Z)

beta = 0.025
print ('Уровень значимости =', beta)

alpha = 1 - beta


k = amount_of_interval - 3
print ('Количество степеней свободы =', k)

critical_chi2 = scipy.stats.chi2.ppf(alpha, df=k)
print('χ² критическое =',critical_chi2)


gamma = 0.9
print('Надежность (γ) =',gamma)

print('Доверительный интервал для математического ожидания:')

t = math.sqrt(2)*scipy.special.erfinv(2*(gamma/2))

print ('2Ф(tγ) = γ, tγ =',t)

M1 = X-t*s/math.sqrt(size)
M2 = X+t*s/math.sqrt(size)
print ('P (',M1,'< M(X) <',M2,') ≥',gamma)

print('Доверительный интервал для исправленного среднего квадратичного отклонения:')

alpha1 = 1 - (1+gamma)/2
print ("α₁ =",alpha1)
alpha2 = 1 -(1-gamma)/2
print ("α₂ =",alpha2)

k = size - 1
print ("k =",k)


critical_chi_1 = scipy.stats.chi2.ppf(alpha2, df=k)
critical_chi_2 = scipy.stats.chi2.ppf(alpha1, df=k)
print("χ₁² критическое при (α₁,k) =", critical_chi_2)
print("χ₂² критическое при (α₂,k) =", critical_chi_1)

s_2 = s*math.sqrt(size-1)/(math.sqrt(critical_chi_2))
s_1 = s*math.sqrt(size-1)/(math.sqrt(critical_chi_1))

print ('P (',s_1,'< s(X) <',s_2,') ≥',gamma)
#print (np.sum(array_y))


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot()
#print(least)
print ('Гистограмма относительных частот')

x = [least+a*length for a in range(0, amount_of_interval)]
x = array_x
y = array_y
ax.bar(x, y, width = length/2, linewidth = 4, edgecolor = 'grey', color = 'aqua' )
ax.grid()
plt.show()
