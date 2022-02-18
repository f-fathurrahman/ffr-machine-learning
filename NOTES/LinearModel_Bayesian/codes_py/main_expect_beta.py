from sympy import *

r = symbols("r", positive=True, real=True)
α, β = symbols("alpha beta", integer=True, positive=True)

#print(α.assumptions0)

p = r**(α - 1) * (1 - r)**(β - 1)
normalization = gamma(α + β)/(gamma(α) * gamma(β))

# quite difficult to integrate for symbolic α and β
#dict_num = {α: 1, β: 1} # scenario 1
dict_num = {α: 2, β: 2}

res = integrate(r*p.subs(dict_num), (r,0,1))
print("normalization = ", normalization.subs(dict_num) )
expect_r = res*normalization.subs(dict_num)
print("expect r  = ", expect_r)

res2 = integrate(r*r*p.subs(dict_num), (r,0,1))
expect_r2 = res2*normalization.subs(dict_num)
print("expect r2 = ", expect_r2 )

var_r = expect_r2 - expect_r**2
print("var_r = ", var_r)

