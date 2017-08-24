from matplotlib import pyplot as plt
plt.switch_backend('Qt5Agg')

import pandas as pd

if __name__ == '__main__':
	df = pd.read_csv('../data/wp_yaw_const.txt')
	df.columns=['x','y','v','yaw']
	plt.plot(df['x'], df['y'])
	plt.show()
	
