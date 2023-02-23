import matplotlib.pyplot as plt
import numpy as np

# x = np.array([0, 1, 2, 3, 4], dtype=np.float32)
# y = np.array([2, 3, 4, 5, 6], dtype=np.float32)
x = np.array([0, 1, 2, 3, 4], dtype=np.float32)
y = np.array([2,5,8,11,14], dtype=np.float32)

def predict(a, xt):
	return a[0]+a[1]*xt

def MSE(a, x, y):
	total = 0
	for i in range(len(x)):
		total += (y[i]-predict(a,x[i]))**2
	return total

def loss(p):
	p=[y[0],p]
	return MSE(p, x, y)

# p = [0.0, 0.0]
# plearn = optimize(loss, p, max_loops=3000, dump_period=1)
def hillClimbing(f, x, dx=0.01):
	while (True):
		#print('x={0:.2f} f(x)={1:.2f}'.format(x, f(x)))
		if f(x+dx)<f(x): # 如果右邊的高度 f(x+dx) > 目前高度 f(x) ，那麼就往右走
			x = x + dx
		elif f(x-dx)<f(x): # 如果左邊的高度 f(x-dx) > 目前高度 f(x) ，那麼就往左走
			x = x - dx
		else: # 如果兩邊都沒有比現在的 f(x) 高，那麼這裡就是區域最高點，直接中斷傳回
			break
	return x

def optimize():
	# 請修改這個函數，自動找出讓 loss 最小的 p
	# p = [2,1] # 這個值目前是手動填的，請改為自動尋找。(即使改了 x,y 仍然能找到最適合的回歸線)
	#p = [3,2] # 這個值目前是手動填的，請改為自動尋找。(即使改了 x,y 仍然能找到最適合的回歸線)
	
	s=hillClimbing(loss,0.1)
	#s=loss(0.1)
	print(s)
	#p=[2,1.1]
	p=[y[0],s]
	return p

p = optimize()




# Plot the graph
y_predicted = list(map(lambda t: p[0]+p[1]*t, x))
print('y_predicted=', y_predicted)
plt.plot(x, y, 'ro', label='Original data')
plt.plot(x, y_predicted, label='Fitted line')
plt.legend()
plt.show()
