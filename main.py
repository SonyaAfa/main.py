import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import scatter

#natural cubic spiline
def ncs(Nodes):
    n=len(Nodes)-1
    a=np.zeros(n+1)
    j=0
    for i in Nodes:
        a[j]=i[1]
        j+=1
    b = np.zeros(n)
    d=np.zeros(n)
    h=np.zeros(n)
    for i in range(n):
        h[i]=Nodes[i+1][0]-Nodes[i][0]
    alpha=np.zeros(n)
    for i in range(1,n,1):
        alpha[i]=3/h[i]*(a[i+1]-a[i])-3/h[i-1]*(a[i]-a[i-1])
    c=np.zeros(n+1)
    l = np.zeros(n + 1)
    mu = np.zeros(n + 1)
    z = np.zeros(n + 1)
    l[0]=1
    mu[0]=0
    z[0]=0
    for i in range(1, n, 1):
        l[i]=2*(Nodes[i+1][0]-Nodes[i-1][0])-h[i-1]*mu[i-1]
        mu[i]=h[i]/l[i]
        z[i]=(alpha[i]-h[i-1]*z[i-1])/l[i]
    l[n]=1
    z[n]=0
    c[n]=0
    for j in range(n-1,-1,-1):
        c[j]=z[j]-mu[j]*c[j+1]
        b[j]=(a[j+1]-a[j])/h[j]-(h[j]*(c[j+1]+2*c[j]))/3
        d[j]=(c[j+1]-c[j])/(3*h[j])
    output_set=np.zeros((n,5))
    for i in range(n):
        output_set[i][0]=a[i]
        output_set[i][1]=b[i]
        output_set[i][2] = c[i]
        output_set[i][3] = d[i]
        output_set[i][4] = Nodes[i][0]
    return(output_set)


def draw_crowbar(points,col):
    FirstCoordinate = []
    SecondCoordinate = []

    for i in range(len(points)):
        FirstCoordinate = FirstCoordinate + [points[i][0]]
        SecondCoordinate = SecondCoordinate + [points[i][1]]
        if i<len(points)-1:
            plt.plot([points[i][0],points[i+1][0]],[points[i][1],points[i+1][1]],color=col)
    # print('coord', FirstCoordinate, SecondCoordinate)
    plt.scatter(FirstCoordinate, SecondCoordinate,color='black')
    #plt.show()
def draw_polygon(corners):
    A, B, C, D = smash_the_bounary(corners)
    draw_crowbar(A,'red')
    draw_crowbar(B,'green')
    draw_crowbar(C,'blue')
    draw_crowbar(D,'gold')
    # print('coord', FirstCoordinate, SecondCoordinate)
    #plt.scatter(FirstCoordinate, SecondCoordinate,color='black')
    #plt.show()


#процудура разбивающая границу выпуклого многоугольника на 4 части
#каждая часть - ломоная. Процедура возвращает матрицу точек(координат) в порядке против часовой стрелки
#работает только для многоугольников в общем положении
def smash_the_bounary(Corners):
    A=[]#
    B=[]
    C=[]
    D=[] #A,B,C,D --- cписки точек
    n=len(Corners)
   # new_corners=np.zeros((n+1,2))
    #for i in range(n):
    #    new_corners[i]=Corners[i]
    #new_corners[n]=Corners[0]
    #abcd=[]
    a=0
    b=0
    i=0
    while a==b:
        x=Corners[i][0]
        y=Corners[i][1]
        x_prev=Corners[(i-1)%n][0]
        y_prev=Corners[(i-1)%n][1]
        x_next=Corners[(i+1)%n][0]
        y_next=Corners[(i+1)%n][1]
        if (x>x_prev and y<y_prev):
            a=1
        if (x>x_prev and y>y_prev):
            a=2
        if (x<x_prev and y>y_prev):
            a=3
        if (x<x_prev and y<y_prev):
            a=4
        if (x_next>x and y_next<y):
            b=1
        if (x_next>x and y_next>y):
            b=2
        if (x_next<x and y_next>y):
            b=3
        if (x_next<x and y_next<y):
            b=4
        if a==b:
            i=i+1%n
    j=0
    a1=a# запомним какое ребро было перед той верщиной с которой начинали
    b1=b
    while j<n:
        x = Corners[i][0]
        y = Corners[i][1]
        x_prev = Corners[(i - 1) % n][0]
        y_prev = Corners[(i - 1) % n][1]
        x_next = Corners[(i + 1) % n][0]
        y_next = Corners[(i + 1) % n][1]
        if (x > x_prev and y < y_prev):
            a = 1
        if (x > x_prev and y > y_prev):
            a = 2
        if (x < x_prev and y > y_prev):
            a = 3
        if (x < x_prev and y < y_prev):
            a = 4
        if (x_next > x and y_next < y):
            b = 1
        if (x_next > x and y_next > y):
            b = 2
        if (x_next < x and y_next > y):
            b = 3
        if (x_next < x and y_next < y):
            b = 4
        if (a==1 and b==1):
            A.append(Corners[i%n])
        if (a==2 and b==2):
            B.append(Corners[i%n])
        if (a==3 and b==3):
            C.append(Corners[i%n])
        if (a==4 and b==4):
            D.append(Corners[i%n])
        if (a==1 and b==2):
            A.append(Corners[i%n])
            B.append(Corners[i%n])
        if (a==1 and b==3):
            A.append(Corners[i%n])
            C.append(Corners[i%n])
        if (a==2 and b==3):
            B.append(Corners[i%n])
            C.append(Corners[i%n])
        if (a==2 and b==4):
            B.append(Corners[i%n])
            D.append(Corners[i%n])
        if (a==3 and b==4):
            C.append(Corners[i%n])
            D.append(Corners[i%n])
        if (a==3 and b==1):
            C.append(Corners[i%n])
            A.append(Corners[i%n])
        if (a==4 and b==1):
            D.append(Corners[i%n])
            A.append(Corners[i%n])
        if (a==4 and b==2):
            D.append(Corners[i%n])
            B.append(Corners[i%n])
        j+=1
        i=(i+1)%n
    if a1==1:
        A.append(Corners[i])
    if a1==2:
        B.append(Corners[i])
    if a1==3:
        C.append(Corners[i])
    if a1==4:
        D.append(Corners[i])

    #if abcd(0)!=abcd(len(abcd)-1):

    A1=np.zeros((len(A),2))#A1,B1,C1,D1 --- матрицы точек
    for i in range(len(A)):
        A1[i][0]=A[i][0]
        A1[i][1]=A[i][1]
    B1 = np.zeros((len(B), 2))
    for i in range(len(B)):
        B1[i][0]=B[i][0]
        B1[i][1]=B[i][1]
    C1 = np.zeros((len(C), 2))
    for i in range(len(C)):
        C1[i][0]=C[i][0]
        C1[i][1]=C[i][1]
    D1 = np.zeros((len(D), 2))
    for i in range(len(D)):
        D1[i]=D[i][0]
        D1[i][1]=D[i][1]
    return A1,B1,C1,D1

def sgn(x):
    if x>0:
        s=1
    if x<0:
        s=-1
    if x==0:
        s=0
    return s


#процудкра определяющая для ребра многоугольника (вершины перечислены против часовой стрелки) A(x1,y1) B(x2,y2)
# лежит точка в той же полуплоскости от прямой содерожащей ребро или нет
#иными словами если идти от А к B процедура возвращает 1 если точка (x,y) в левой полуплоскости, -1 если в правой
# 1- лежит
#-1 - не лежит
#0 лежит на границе именно на ребре
#2 лежит на прямой но не на ребре
#3 --- A=B
def in_half_plane(A, B, x, y):
    if A[0]==B[0] and A[1]==B[1]:
        print('A=B')
        b=3
    else:
        if A[0]==B[0]:
            if x<A[0]:
                b=1
            if x>A[0]:
                b=-1
            if x==A[0]:
                if (y<B[1]) and (y>A[1]):
                    b=0
                else:
                    b=2
        else:
            if A[1]==B[1]:
                if y<A[1]:
                    b=-1
                if y>A[1]:
                    b=1
                if y==A[1]:
                    if (x<B[0]) and (x>A[0]):
                        b=0
                    else:
                        b=2
            else:
                b=-sgn((x-A[0])*(B[1]-A[1])-(B[0]-A[0])*(y-A[1]))
                if b==0  and ((x<min(A[0],B[0])) or (x>max(A[0],B[0])) or y<min(A[1],B[1]) or y>max(A[1],B[1])):
                    b=2
    return b

#процедура определяющая лежит ли точка (x,y) внутри многоугольника
#1 - лежит
#-1 не лежит
#0 - лежит на границе
#предполагается что вершины многоугольника перечислены против часовой стрелки
def in_polygon(Corners,x,y):
    new_corners=np.zeros((len(Corners)+1,2))
    for i in range(len(Corners)):
        new_corners[i]=Corners[i]
    new_corners[len(Corners)]=Corners[0]
    b=1
    for i in range(len(Corners)):
        if in_half_plane(new_corners[i],new_corners[i+1],x,y)==0:
            b=0
        if in_half_plane(new_corners[i],new_corners[i+1],x,y)==-1:
            b=-1
        if in_half_plane(new_corners[i], new_corners[i + 1], x, y) == 2:
            b=-1
    return b

#процедура возвращающая точку пересечения прямых
def line_intersection(line1, line2):
    x=0
    y=0

    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]) #Typo was here

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       print('lines do not intersect')
    else:
        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
    return x, y

#процедура определяющая axis-parallel прямоугольник наибольшей площади, у которого противоположные углы ледат
# на сегментах segment1, segment2
def maxS_axis_parallel_rectangle_with_two_corners_on_segments(segment1,segment2):
    S=0
    x1=segment1[0][0]
    y1=segment1[0][1]
    x2=segment1[1][0]
    y2=segment1[1][1]
    x3=segment2[0][0]
    y3=segment2[0][1]
    x4=segment2[1][0]
    y4=segment2[1][1]
    x5=2*x4-x3
    y5=y3
    x6=x1+x4-x5
    y6=y1+y4-y5
    x7=x2+x4-x5
    y7=y2+y4-y5
    line1=np.zeros((2,2))
    #line2 = np.zeros((2, 2))
    e1=[x1,y1]
    e2=[x3,y3]
    line1[0]=[x1,y1]
    line1[1]=[x6,y6]
    line2=segment2
    a,b=line_intersection(line1,line2)
    #if (in_half_plane(segment2[0],segment2[1],a,b)==0):
    if (a >= min(line2[0][0], line2[1][0])
         and a <= max(line2[0][0], line2[1][0])
         and b >= min(line2[0][1], line2[1][1])
         and b<=max(line2[0][1],line2[1][1])):
        e2=[a,b]
        S=abs((a-x1)*(b-y1))
    print(a,b)
    line1[0] = [x2, y2]
    line1[1] = [x7, y7]
    a, b = line_intersection(line1, line2)
    #if (in_half_plane(segment2[0], segment2[1], a, b) == 0):
    if (a>=min(line2[0][0],line2[1][0])
        and a<=max(line2[0][0],line2[1][0])
        and b>=min(line2[0][1],line2[1][1])
        and b<=max(line2[0][1],line2[1][1])):
        if abs((a-x2)*(b-y2))>S:
            S=abs((a-x2)*(b-y2))
            e1=[x2,y2]
            e2=[a,b]

    print(a,b)
    x8=2*x2-x1
    y8=y1
    line2[0]=[x3,y3]
    line2[1]=[x3+x2-x8,y3+y2-y8]
    a, b = line_intersection(segment1, line2)
    #if (in_half_plane(segment1[0], segment1[1], a, b) == 0):
    if (a >= min(segment1[0][0], segment1[1][0])
            and a <= max(segment1[0][0], segment1[1][0])
            and b >= min(segment1[0][1], segment1[1][1])
            and b <= max(segment1[0][1], segment1[1][1])):
        if abs((a - x3) * (b - y3)) > S:
            S = abs((a - x3) * (b - y3))
            e1 = [x3, y3]
            e2 = [a, b]
    line2[0] = [x4, y4]
    line2[1] = [x4 + x2 - x8, y4 + y2 - y8]
    a, b = line_intersection(segment1, line2)
    #if (in_half_plane(segment1[0], segment1[1], a, b) == 0):
    if (a>=min(segment1[0][0],segment1[1][0])
        and a<=max(segment1[0][0],segment1[1][0])
        and b>=min(segment1[0][1],segment1[1][1])
        and b<=max(segment1[0][1],segment1[1][1])):
        if abs((a - x4) * (b - y4)) > S:
            S = abs((a - x4) * (b - y4))
            e1 = [x4, y4]
            e2 = [a, b]
    return e1,e2,S#противоположные углы прямоугольника(на сегментах) и его площадь

#ghоцудура возвращающая прямлйгольник наибольшей площади среди тех
#которые имеют ровно 2 угла (на A,C) на границе
def maxS_par_rec_with_two_cor_on_crow(A,C,Corners):
    #AC
    n=len(A)-1
    m=len(C)-1
    S=0
    l1='none'
    l3='none'
    for i in range(n):
        for j in range(m):
            segment1=np.zeros((2,2))
            segment2 = np.zeros((2, 2))
            segment1[0]=A[i]
            segment1[1]=A[i+1]
            segment2[0]=C[j]
            segment2[1]=C[j+1]
            e1,e3,s=maxS_axis_parallel_rectangle_with_two_corners_on_segments(segment1,segment2)
            if in_polygon(Corners,e3[0],e1[1])==1 and in_polygon(Corners,e1[0],e3[1])==1:
                if s>S:
                    S=s
                    l1=e1
                    l3=e3
    return l1,l3,S
#процудура определяющая прямоугольник наибольшей площади среди тех которые вписаны в многоугольник с вершинами corners
#(перечисленными против часовой стрелки)
# имеющих ровно два угла (а значит противоположных) на границах многоугольника
def maxS_parall_axis_rect_with_two_corners_on_the_boundary(Corners):
    A,B,C,D=smash_the_bounary(Corners)
    #AC
    S='0'
    l1,l3,Sac=maxS_par_rec_with_two_cor_on_crow(A,C,Corners)
    e1, e3, Sbd = maxS_par_rec_with_two_cor_on_crow(B, D,Corners)
    if Sbd>Sac:
        m1=e1
        m2=e3
        S=Sbd
    else:
        m1=l1
        m2=l3
        S=Sac
    return l1,l3,S

def intersect_line_with_crow(line,crow):
    b=False
    number_of_intersected_edge=1
    n=len(crow)-1
    line2=np.zeros((2,2))
    x1=0
    y1=0
    for i in range(n):
        line2[0]=crow[i]
        line2[1]=crow[i+1]
        x,y=line_intersection(line,line2)
        if x>=min(line2[0][0],line2[1][0]) and x<=max(line2[0][0],line2[1][0]):
            b=True
            x1=x
            y1=y
           # number_of_intersected_edge=i
    return x1,y1,b,number_of_intersected_edge

#процудура возвращающая прямлйгольник наибольшей площади среди тех
#которые имеют ровно 2 угла (на A,B,C) на границе, причем ни один угол прямоугольника не совпадает с углом многоугольника
def maxS_par_rec_with_three_cor_on_crow(A,B,C,Corners):
    n = len(A) - 1
    m = len(C) - 1
    k=len(B)-1
    S = 0
    line1 = np.zeros((2, 2))
    line2 = np.zeros((2, 2))
    line3 = np.zeros((2, 2))
    e1=np.zeros(2)
    e3=np.zeros(2)
    b1=False
    b2=False
    b3=False
    if n>0 and m>0 and k>0:#то есть никакое из множеств A,B,C не пустое
        for i in range(n):
            for j in range(k):
                for l in range(m):
                    line1[0] = A[i]
                    line1[1] = A[i + 1]
                    line2[0] = B[j]
                    line2[1] = B[j + 1]
                    line3[0] = C[l]
                    line3[1] = C[l + 1]
                    # прямоугольник у которого ни одна вершина не совпадает с углом многоугольника
                    x1, y1 = line_intersection(line1, line2)
                    x2, y2 = line_intersection(line2, line3)
                    x21 = (x1 + x2) / 2
                    y21 = (y1 + y2) / 2
                    b2 = (x21 >= min(B[j][0],B[j+1][0]) and x21 <=max(B[j][0], B[j + 1][0])) # середина отрезка попала на сегмент
                    xx21 = np.zeros((2, 2))
                    xx21[0] = [x21, y21]
                    xx21[1] = [x21, y21 + 1]
                    x31, y31 = line_intersection(line3, xx21)
                    b3 = (x31 >= min(C[l][0],C[l+1][0]) and x31 <= max(C[l + 1][0],C[l][0]))  #

                    yy21 = np.zeros((2, 2))
                    yy21[0] = [x21, y21]
                    yy21[1] = [x21 + 1, y21]
                    x11, y11 = line_intersection(line1, yy21)
                    b1 = (x11 >= min(A[i][0],A[i+1][0]) and x11 <= max(A[i][0],A[i + 1][0]))

                    b4=in_polygon(Corners,x11,y31)
                    s = abs((x21 - x11) * (y31 - y21))

                    if b1 and b2 and b3 and (b4>=0) and s>S:
                        e1 = [x11, y11]
                        e3 = [x31, y31]
                        S = s
    #перебор случаев когда одна их вершин прямоугольника совпадает с углом многоугольника

    return e1,e3,S

#процедура, возвращающая список вершин без k-й точки, нумируя начиная с k+1
def delete_point(Samples,k):
    n=len(Samples)
    new_crow = np.zeros((n - 1, 2))
    j=k+1
    for i in range(n-1):
        new_crow[i]=Samples[j%n]
        j=(j+1)%n
    return new_crow

#процедура, возвращающая список вершин, нумируя начиная с k+1
def delete_edge(Samples,k):#
    n=len(Samples)
    new_crow = np.zeros((n, 2))
    for i in range(n):
        new_crow[i]=Samples[(i+k+1)%n]
    return new_crow

#процедура нахождения прямоугольника макс площади среди техз у которых одна из вершин совпадает с углом многоугольника и
# 3 вершины на границе многоугольника
def maxS_par_rec_with_three_cor_on_crow_and_point(Corners):
    A, B, C, D = smash_the_bounary(Corners)
    S=0
    line1 = np.zeros((2, 2))
    line2 = np.zeros((2, 2))
    line3 = np.zeros((2, 2))
    e1 = np.zeros(2)
    e3 = np.zeros(2)
    n=len(Corners)
    Broken_corners=np.zeros(n-1)
    for i in range(n):
        x0=Corners[i][0]
        y0=Corners[i][1]
        line1[0]=Corners[i]
        line1[1]=[x0+1,y0]#horizontal line y=y0
        line2[0] = Corners[i]
        line2[1] = [x0 , y0+1]  # вертикальная line x=x0
        broken_corners =delete_point(Corners,i)
        x1,y00,b1,num1=intersect_line_with_crow(line1,broken_corners)
        x00,y1,b2,m=intersect_line_with_crow(line2,broken_corners)
        s=abs((x1-x0)*(y1-y0))
        if b1 and b2 and in_polygon(Corners,x1,y1)>=0 and s>S:
           S=s
           e1=[x0,y0]
           e3=[x1,y1]
           print(e1,e3,s)
            #
        line3[0] = [x1,y0]
        line3[1] = [x1 , y0+1]  # вертикальная line x=x1
        new_breakx1y0=delete_edge(Corners,num1)
        x1,y1,b3,k=intersect_line_with_crow(line3,new_breakx1y0)
        s = abs((x1 - x0) * (y1 - y0))
        if b1 and b3 and in_polygon(Corners,x0,y1)>=0 and s>S:
           S=s
           e1=[x0,y0]
           e3=[x1,y1]
           print('e1e3s',e1,e3,s)

    return e1,e3,S
def main():
    nodes=np.loadtxt('Nodes')  # читаю данные из файла как матрицу
    #print(nodes, ncs(nodes))
    #my=open('Corners','w')
    #my.close()
    corners = np.loadtxt('Corners')  # читаю данные из файла как матрицу
    draw_polygon(corners)
    plt.show()
    A,B,C,D=smash_the_bounary(corners)
    print('a',A)
    print('b', B)
    print('C', C)
    print('D', D)
    line1=np.zeros((2,2))
    line1[0]=[0,0]
    line1[1]=[1,1]
    line2 = np.zeros((2, 2))
    line2[0] = [2, 3]
    line2[1] = [3, -2]


    A=np.zeros(2)
    A[0]=10
    A[1]=0
    B = np.zeros(2)
    B[0]=10
    B[1]=1
    x=10
    y=0
    print('inhalfplane',in_half_plane(A,B,x,y))
    print('inpolygon',in_polygon(corners,0,-8))

#поиск прямоугольника с двумя углами на сегментах
    A, B, C, D = smash_the_bounary(corners)
    segment1=np.zeros((2,2))
    segment2 = np.zeros((2, 2))
    segment1[1]=[3,-1]
    segment2[0]=[5,3]
    segment2[1]=[1,6]
    e1,e2,s=maxS_axis_parallel_rectangle_with_two_corners_on_segments(segment1,segment2)
    print('maxRL',e1,e2,s)
    rectangle=np.zeros((4,2))
    rectangle[0]=e1
    rectangle[1][0]=e2[0]
    rectangle[1][1]=e1[1]
    rectangle[2]=e2
    rectangle[3][0]=e1[0]
    rectangle[3][1]=e2[1]
    #draw_crowbar(rectangle,'blue')
    #plt.show()
    #поиск прямоугольника с двумя углами на границах
    l1,l3,s=maxS_parall_axis_rect_with_two_corners_on_the_boundary(corners)
    print('maxR2',l1,l3,s)
    draw_polygon(corners)
    if s!=0:
        rectangle = np.zeros((5, 2))
        rectangle[0] = l1
        rectangle[1][0] = l3[0]
        rectangle[1][1] = l1[1]
        rectangle[2] = l3
        rectangle[3][0] = l1[0]
        rectangle[3][1] = l3[1]
        rectangle[4] = l1

        draw_crowbar(rectangle,'black')
        #plt.show()
    else:
        print('s=0')
    l1, l3, s=maxS_par_rec_with_three_cor_on_crow(A,B,C,corners)
    l1, l3, s = maxS_par_rec_with_three_cor_on_crow( B, C,D, corners)
    l1, l3, s = maxS_par_rec_with_three_cor_on_crow( C, D,A, corners)
    l1, l3, s = maxS_par_rec_with_three_cor_on_crow( D, A,B, corners)
    if s!=0:
        rectangle = np.zeros((5, 2))
        rectangle[0] = l1
        rectangle[1][0] = l3[0]
        rectangle[1][1] = l1[1]
        rectangle[2] = l3
        rectangle[3][0] = l1[0]
        rectangle[3][1] = l3[1]
        rectangle[4] = l1

        draw_crowbar(rectangle,'pink')
        #plt.show()
    else:
        print('s=0')


    l1,l3,s=maxS_par_rec_with_three_cor_on_crow_and_point(corners)
    if s!=0:
        rectangle = np.zeros((5, 2))
        rectangle[0] = l1
        rectangle[1][0] = l3[0]
        rectangle[1][1] = l1[1]
        rectangle[2] = l3
        rectangle[3][0] = l1[0]
        rectangle[3][1] = l3[1]
        rectangle[4] = l1

        draw_crowbar(rectangle,'blue')
        #plt.show()
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
    print('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
