import random

xy = [(0,0), (1,0), (2,0), 
      (3,0), (3,1), (3,2), 
      (3,3), (2,3), (1,3), 
      (0,3), (0,2), (0,1)]
s0 = [1,2,3,4,5,6,7,8,9,10,11,0]
s = [0,1,2,3,4,5,6,7,8,9,10,11]
random.shuffle(s)

def distance(a,b):
    dx = a[0]-b[0]
    dy = a[1]-b[1]
    return (dx*dx+dy*dy)**0.5

def circle_length(s):
    d = 0.0
    for i in range(len(s)):
        d += distance(xy[i], xy[s[i]])
    return d

print('s0=', s0)
print('circle_length(s0)=', circle_length(s0))

r=[]
j=[]
for i in range(3000):
    random.shuffle(s)
    #print(l)
    #print('s=', s)
    j.append(s)
    a=circle_length(s)
    #print('circle_length(s)=', a)
    r.append(a)

min_value = min(r)
print("min_value=",min_value)
#print(r.index(min_value))
print(j[r.index(min_value)])
#print('s=', s)
#print('circle_length(s)=', circle_length(s))