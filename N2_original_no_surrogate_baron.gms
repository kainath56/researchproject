*Title: 'Select 2 solvents to maximise the solubility of Ibuprofen
*N=2
*Solved with DICOPT
*Miscibility constraints are included
*==============================================================================
SETS

s 'solvents'
/methanol, ethanol, 2-propanol, acetone, mibk, ethylacetate,
chloroform, toluene, water/

i 'components in mixture'
/ibuprofen, s1, s2/

ii(i) 'selected solvent'
/s1, s2/


k 'groups'
/ch3, ch2, ch, ach, acch3, acch2, acch, oh, ch3oh, h2o, ch3co, ch3coo, cooh, chcl3/

c 'integer cuts'
/1*5/

dyn(c) 'dynamic set of c';

alias(k,m);
alias(k,p);
alias(s,ss);


PARAMETERS
Hm 'enthalpy of fusion of ibuprofen' /25500/
Tm 'melting point of ibuprofen' /347.15/
T 'system temperature' /300/
Rg 'gas constant' /8.3144/


R(k) 'Van der Waals volume of group k'
/ch3             0.9011
ch2              0.6744
ch               0.4469
ach              0.5313
acch3            1.2663
acch2            1.0396
acch             0.8121
oh               1.0000
ch3oh            1.4311
h2o              0.9200
ch3co            1.6724
ch3coo           1.9031
cooh             1.3013
chcl3            2.8700 /

Q(k) 'Van der Waals surface area of group k'
/ch3             0.848
ch2              0.540
ch               0.228
ach              0.400
acch3            0.968
acch2            0.660
acch             0.348
oh               1.200
ch3oh            1.432
h2o              1.400
ch3co            1.488
ch3coo           1.728
cooh             1.224
chcl3            2.410 /


table v(s,k) 'number of groups in molecule i'
                 ch3     ch2     ch      ach     acch3   acch2   acch    oh      ch3oh   h2o     ch3co   ch3coo  cooh    chcl3

methanol                                                                         1
ethanol          1       1                                               1
2-propanol       2               1                                       1
acetone          1                                                                               1
mibk             2       1       1                                                               1
ethylacetate     1       1                                                                               1
chloroform                                                                                                               1
toluene                                  5       1
water                                                                                    1


table vib(i,k)'identity of ibuprofen'
             ch3     ch ach      acch2 acch   cooh
ibuprofen     3       1  4         1    1       1      ;


                                                                                                                          ;
*===============================================================================
*group interaction parameters
*===============================================================================
table a(k,m) 'group interaction parameters'
         ch3     ch2     ch      ach     acch3   acch2   acch    oh      ch3oh   h2o     ch3co   ch3coo  cooh    chcl3
ch3      0       0       0       61.13   76.5    76.5    76.5    986.5   697.2   1318    476.4   232.1   663.5   24.9
ch2      0       0       0       61.13   76.5    76.5    76.5    986.5   697.2   1318    476.4   232.1   663.5   24.9
ch       0       0       0       61.13   76.5    76.5    76.5    986.5   697.2   1318    476.4   232.1   663.5   24.9
ach      -11.12  -11.12  -11.12  0       167     167     167     636.1   637.4   903.8   25.77   5.994   537.4   -231.9
acch3    -69.7   -69.7   -69.7   -146.8  0       0       0       803.2   603.3   5695    -52.1   5688    872.3   -80.25
acch2    -69.7   -69.7   -69.7   -146.8  0       0       0       803.2   603.3   5695    -52.1   5688    872.3   -80.25
acch     -69.7   -69.7   -69.7   -146.8  0       0       0       803.2   603.3   5695    -52.1   5688    872.3   -80.25
oh       156.4   156.4   156.4   89.6    25.82   25.82   25.82   0       -137.1  353.5   84      101.1   199     -98.12
ch3oh    16.51   16.51   16.51   -50     -44.5   -44.5   -44.5   249.1   0       -181    23.39   -10.72  -202    -139.4
h2o      300     300     300     362.3   377.6   377.6   377.6   -229.1  289.6   0       -195.4  72.87   -14.09  353.7
ch3co    26.76   26.76   26.76   140.1   365.8   365.8   365.8   164.5   108.7   472.5   0       -213.7  669.4   -354.6
ch3coo   114.8   114.8   114.8   85.84   -170    -170    -170    245.4   249.6   200.8   372.2   0       660.2   -209.7
cooh     315.3   315.3   315.3   62.32   89.86   89.86   89.86   -151    339.8   -66.17  -297.8  -256.3  0       39.63
chcl3    36.7    36.7    36.7    288.5   69.9    69.9    69.9    742.1   649.1   826.8   552.1   176.5   504.2   0
                                                                                                                    ;



parameter ps(k,m);
ps(k,m)= exp(-a(k,m)/T);
Display ps;
parameter nib(i,k);
nib('ibuprofen',k)= vib('ibuprofen',k);
parameter qib;
qib = sum(k,nib('ibuprofen',k)*Q(k));
parameter rib;
rib = sum(k,nib('ibuprofen',k)*R(k));
parameter qs(s);
qs(s) = sum(k,v(s,k)*Q(k));
parameter rs(s);
rs(s) = sum(k,v(s,k)*R(k));

option decimals=4;
Display qs,rs;

parameter eib(i,k);
eib('ibuprofen',k)=vib('ibuprofen',k)*Q(k)/qib;
parameter bib(i,k);
bib('ibuprofen',k)=sum(m,eib('ibuprofen',m)*ps(m,k));
parameter yv(ii,s,c) 'store y values from previous iterations';
parameter zv(c) 'store objective values from previous iterations';
parameter xv(i,c) 'store mole fractions from previous iterations';


POSITIVE VARIABLES

x(i)             'liquid phase mole fraction of component i'
rc(i)            'van der waals volume of component i'
qc(i)            'van der waals area of component i'
J(i),L(i)

b(i,k)          'intermediates'
th(k)
w(k)

xm(ii)
th12(k)
w12(k)
sumr12
sumq12
b12(ii,k)

FREE VARIABLES
z            'objective function'

lng(i)       'natural log of activity coefficient of ibuprofen'
lngc(i)      'natural log of combinatorial activity coefficient of ibuprofen'
lngr(i)      'natural log of residual activity coefficient of ibuprofen'

d_lngc12(ii)  'differention of combinatorial act_coeff'
d_lngr12(ii)  'differention of residual act_coeff'
d_lng12(ii)   'differentiation of act_coef wrt to x(ii)'
d_th12(k)     'diff of intermediate'
d_w12(k)      'diff of intermediate'
a12(k)     'intermediate'
c12(k)     'intermediate'



BINARY VARIABLES
y(ii,s)     'selected solvents'

INTEGER VARIABLE
n(i,k)    'number of groups k in component i' ;


EQUATIONS
eq_z                   'objective function maximise solubility of ibuprofen'
eq_x                   'mole fraction constrain'
eq_n(ii,k)
eq_qc(ii)
eq_rc(ii)
eq_J,eq_L
eq_lngc
eq_th(k)
eq_w(k)
eq_lngr
eq_lng
eq_solub
*equations for logical conditions
logic1(ii),logic2(s)
logic3(s,ss)
*equations for miscibility constraints
eq_xm1,eq_xm2
eq_sumr12,eq_sumq12,eq_dlngc12,eq_th12(k),eq_dth12(k),eq_w12(k),eq_dw12(k),eq_a12(k),eq_b12(ii,k),eq_c12(k),eq_dlngr12,eq_dlng12,eq_misc12
IntCut(c)
;



*********************choose only 2 solvents*************************************
logic1(ii)..
sum(s,y(ii,s))=e=1;

**********************each solvent at most once*********************************
logic2(s)..y('s1',s)+y('s2',s)=l=1;

*****************************solvent ordering***********************************
logic3(s,ss)$(ord(ss) <ord(s))..y('s1',s)+y('s2',s-ord(ss))=l=1;



*******************************UNIFAC*******************************************
eq_z..
z=e=x('ibuprofen');

eq_x..
sum(i,x(i))=e=1;

eq_n(ii,k)..
n(ii,k)=e=sum(s,v(s,k)*y(ii,s));


eq_qc(ii)..
qc(ii)=e= sum(s,qs(s)*y(ii,s));

eq_rc(ii)..
rc(ii)=e= sum(s,rs(s)*y(ii,s));

*combinatorial part of activity coefficient
eq_J..J('ibuprofen')*sum(i,x(i)*rc(i))=e=rib;
eq_L..L('ibuprofen')*sum(i,x(i)*qc(i))=e=qib;

eq_lngc..
lngc('ibuprofen')=e=1-J('ibuprofen')+log(J('ibuprofen'))-5*qib*(1-J('ibuprofen')/L('ibuprofen')+log(J('ibuprofen'))-log(L('ibuprofen')));

*residual part of activity coefficient
eq_th(k)..th(k)*sum(i,x(i)*qc(i))=e=sum(i,x(i)*Q(k)*n(i,k));
eq_w(k)..w(k)=e=sum(m,th(m)*ps(m,k));

eq_lngr..
lngr('ibuprofen')=e=qib*(1-sum(k,th(k)*bib('ibuprofen',k)/w(k)-eib('ibuprofen',k)*(log(bib('ibuprofen',k))-log(w(k)))));

*activity coefficient
eq_lng..
lng('ibuprofen') =e=  lngc('ibuprofen') + lngr('ibuprofen');

*ibuprofen solubility constraint
eq_solub..
log(x('ibuprofen'))+ lng('ibuprofen') =e= Hm/Rg *(1/Tm-1/T);

*****************Miscibility constraints between s1 & s2************************
eq_xm1..xm('s1')*(x('s1')+x('s2'))=e=x('s1');
eq_xm2..xm('s1')+xm('s2')=e=1;

eq_sumr12..sumr12=e=xm('s1')*rc('s1')+(1-xm('s1'))*rc('s2');
eq_sumq12..sumq12=e=xm('s1')*qc('s1')+(1-xm('s1'))*qc('s2');

*differ of combinatorial activity coefficient, i.e. dlngc/dx(s1)
eq_dlngc12..d_lngc12('s1')*power(sumr12,2)*sumq12=e=5*qc('s1')*(qc('s2')-qc('s1'))*power(sumr12,2)
                                        +(rc('s1')-rc('s2'))*(5*qc('s1')-1)*sumr12*sumq12
                                        +5*rc('s1')*(qc('s1')-qc('s2'))*sumr12*sumq12
                                        +((rc('s1')-rc('s2'))*(rc('s1')-5*rc('s1')*(sumq12)))*sumq12;

eq_th12(k)..
th12(k)*sumq12=e=xm('s1')*Q(k)*n('s1',k)+(1-xm('s1'))*Q(k)*n('s2',k);

eq_w12(k)..w12(k)=e=sum(m,th12(m)*ps(m,k));

eq_dth12(k)..
d_th12(k)*power(sumq12,2)=e=((Q(k)*n('s1',k)- Q(k)*n('s2',k))*sumq12)-(xm('s1')*Q(k)*n('s1',k)+ xm('s2')*Q(k)*n('s2',k))*(qc('s1')-qc('s2'));

eq_dw12(k)..d_w12(k)=e=sum(m,ps(m,k)*d_th12(m));

eq_a12(k)..a12(k)*power(w12(k),2)=e=d_th12(k)*w12(k)-th12(k)*d_w12(k);
eq_b12(ii,k)..b12(ii,k)=e=sum(m,Q(m)*n(ii,m)*ps(m,k)) ;
eq_c12(k)..c12(k)*w12(k)=e=d_w12(k);

*differ of residual activity coefficient, i.e. dlngr/dx(s1)
*simplier form, not the general form of unifac
eq_dlngr12..d_lngr12('s1')=e=-sum(k,a12(k)*b12('s1',k)+Q(k)*n('s1',k)*c12(k));

*differ of activity coefficient,i.e. dlng/dx(s1)
eq_dlng12..d_lng12('s1')=e= d_lngc12('s1')+d_lngr12('s1');

eq_misc12..d_lng12('s1')*xm('s1')+1=g=0;

*Integer cuts equation
IntCut(c)$(dyn(c)).. sum((ii,s),yv(ii,s,c)*y(ii,s))-sum((ii,s),(1-yv(ii,s,c))*y(ii,s))=l=sum((ii,s),yv(ii,s,c))-1;

*******************************BOUNDS*******************************************
*bounds on optmization variables
x.up(i)=1;
x.lo(i)=0.01;

n.up(ii,k)=6;

*variable bounds
rc.lo(i)= 0.1;
rc.up(i)= 10;
qc.lo(i)= 0.1;
qc.up(i)= 10;

J.lo('ibuprofen')=0.1;
J.up('ibuprofen')=10;
L.lo('ibuprofen')=0.1;
L.up('ibuprofen')=10;

th.up(k)=1;
th.lo(k)=0;
w.up(k)=3;
w.lo(k)=0;

lngc.lo('ibuprofen')=-30;
lngc.up('ibuprofen')=30;
lngr.lo('ibuprofen')=-30;
lngr.up('ibuprofen')=30;
lng.lo('ibuprofen') = -30;
lng.up('ibuprofen') = 30;

*bounds for imiscibility calculations
*s1 & s2 bounds
*================================================================================
xm.up(ii) = 1;
xm.lo(ii) = 0.01;
th12.up(k)=1;
th12.lo(k)=0;
w12.up(k)=5;
w12.lo(k)=0.0001;

sumr12.lo=0.1;
sumr12.up=10;
sumq12.lo=0.1;
sumq12.up=10;

d_lngc12.lo('s1')=-30;
d_lngc12.up('s1')=30;

d_th12.lo(k)=-30;
d_th12.up(k)=30;

d_w12.lo(k)=-30;
d_w12.up(k)= 30;

a12.lo(k)=-30;
a12.up(k)= 30;

b12.up(ii,k)=30 ;

c12.lo(k)=-30;
c12.up(k)= 30;

d_lngr12.lo('s1')=-30;
d_lngr12.up('s1')=30;

d_lng12.lo('s1')=-30;
d_lng12.up('s1')=30;


*********************INITIAL POINTS*********************************************

*fixing of ibuprofen identity
qc.fx('ibuprofen')=qib;
rc.fx('ibuprofen')=rib;
n.fx('ibuprofen',k)=vib('ibuprofen',k);

*initilaization  of optimal solvent
x.l('ibuprofen')=0.3;
x.l('s2')=0.3;
x.l('s1')=1-x.l('ibuprofen')-x.l('s2');

*y.l('s1','chloroform')=1;
*y.l('s2','water')=1;

y.l('s1','ethylacetate')=1;
y.l('s2','chloroform')=1;

z.l=x.l('ibuprofen');

n.l(ii,k)=sum(s,v(s,k)*y.l(ii,s));

rc.l(ii)= sum(s,rs(s)*y.l(ii,s));
qc.l(ii)= sum(s,qs(s)*y.l(ii,s));

J.l('ibuprofen')=rib/sum(i,x.l(i)*rc.l(i));
L.l('ibuprofen')=qib/sum(i,x.l(i)*rc.l(i));

lngc.l('ibuprofen')=1-J.l('ibuprofen')+log(J.l('ibuprofen'))-5*qib*(1-J.l('ibuprofen')/L.l('ibuprofen')+log(J.l('ibuprofen'))-log(L.l('ibuprofen')));

th.l(k)=sum(i,x.l(i)*Q(k)*n.l(i,k))/sum(i,x.l(i)*qc.l(i));
w.l(k)=sum(m,th.l(m)*ps(m,k));

lngr.l('ibuprofen')=qib*(1-sum(k,th.l(k)*bib('ibuprofen',k)/w.l(k)-eib('ibuprofen',k)*(log(bib('ibuprofen',k))-log(w.l(k)))));

lng.l('ibuprofen') = lngc.l('ibuprofen') + lngr.l('ibuprofen');

*initialisation of imiscibility calculations(s1,s2)
xm.l('s1')=x.l('s1')/(x.l('s1')+x.l('s2'));
xm.l('s2')=1-xm.l('s1');

sumr12.l=xm.l('s1')*rc.l('s1')+(1-xm.l('s1'))*rc.l('s2');
sumq12.l=xm.l('s1')*qc.l('s1')+(1-xm.l('s1'))*qc.l('s2');

d_lngc12.l('s1')=5*qc.l('s1')*(qc.l('s2')-qc.l('s1'))/sumq12.l
                  +(rc.l('s1')-rc.l('s2'))*(5*qc.l('s1')-1)/sumr12.l
                  +5*rc.l('s1')*(qc.l('s1')-qc.l('s2'))/sumr12.l
                  +((rc.l('s1')-rc.l('s2'))*(rc.l('s1')-5*rc.l('s1')*(sumq12.l)))/power(sumr12.l,2);

th12.l(k)=(xm.l('s1')*Q(k)*n.l('s1',k)+(1-xm.l('s1'))*Q(k)*n.l('s2',k))/sumq12.l;

w12.l(k)=sum(m,th12.l(m)*ps(m,k));

d_th12.l(k)=((Q(k)*n.l('s1',k)- Q(k)*n.l('s2',k))*sumq12.l-(xm.l('s1')*Q(k)*n.l('s1',k)+ xm.l('s2')*Q(k)*n.l('s2',k))*(qc.l('s1')-qc.l('s2')))/power(sumq12.l,2);

d_w12.l(k)=sum(m,ps(m,k)*d_th12.l(m));

a12.l(k)=(d_th12.l(k)*w12.l(k)-th12.l(k)*d_w12.l(k))/power(w12.l(k),2);
b12.l(ii,k)=sum(m,Q(m)*n.l(ii,m)*ps(m,k));
c12.l(k)=d_w12.l(k)/w12.l(k);

d_lngr12.l('s1')=-sum(k,a12.l(k)*b12.l('s1',k)+Q(k)*n.l('s1',k)*c12.l(k));

d_lng12.l('s1')=d_lngc12.l('s1')+d_lngr12.l('s1');

MODEL N2 / ALL/;

option nlp=conopt3;
option mip=cplex;
option rminlp=conopt3;
option minlp=baron;

option decimals=5;
OPTION OPTCA = 1e-9;
option iterlim = 1000000;
option optcr  = 1e-4;
option limrow = 0;
option limcol = 0;
OPTION reslim = 10800;

option sysout=on;
dyn(c) = no;
yv(ii,s,c) = 0;
alias (c,cc);
parameter
stab12(c)     'stabbility criterion'
inv1(c);


loop(cc,
    solve N2 using minlp maximising z;
    yv(ii,s,cc) = y.l(ii,s);
    zv(cc) = z.l;
    xv(i,cc) = x.l(i);
    inv1(cc)= 1/xm.l('s1');
    stab12(cc)= d_lng12.l('s1')+ inv1(cc);
    dyn(cc) = yes;


);


display
zv
yv
xv
*stab12
;
