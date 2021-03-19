function x1 = equationsolver0b (a)
x0=a;
%a=5;
%f=@(x)(exp(x) - a*x -1);
%f1=@(x)(exp(x)-a) ;
f=@(x)(exp(x) - 1 -a*x);
f1=@(x)(exp(x) -a) ;
x1=x0-feval(f,x0)/feval(f1,x0);
while (abs(x1-x0)>10^(-3))
    x0=x1;
    x1=x0-feval(f,x0)/feval(f1,x0);
end
%x1;