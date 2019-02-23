import matplotlib.pyplot as plt
import numpy as np

def getEvenIndex(m):
	m_new = np.zeros(shape=(int(m.shape[0]/2), m.shape[1]))
	index = 0
	for i in range(0, m.shape[0]):
		if i % 2 == 0:
			m_new[index, :] = m[i,:]
			index = index + 1
	return m_new
plt.figure(figsize=(8,8))

plt.grid(color='k', linestyle='--', linewidth=0.1)
sigma_cora = np.loadtxt( 'p2_change_cora.txt', delimiter=',')
plt.plot(sigma_cora[:,0], sigma_cora[:,1], '--r', label='Cora', linewidth=10, markersize=10)

sigma_citeseer = np.loadtxt( 'p2_change_citeseer.txt', delimiter=',')
sigma_citeseer = getEvenIndex(sigma_citeseer)
plt.plot(sigma_citeseer[:,0], sigma_citeseer[:,1], 'bo', label='Citeseer', linewidth=10, markersize=13)

sigma_pubmed = np.loadtxt( 'p2_change_pubmed.txt', delimiter=',')
sigma_pubmed = getEvenIndex(sigma_pubmed)
plt.plot(sigma_pubmed[:,0], sigma_pubmed[:,1], 'gs', label='Pubmed', linewidth=10, markersize=13)


#plt.legend(loc='lower left', fontsize=30)
plt.xticks(np.arange(0.4, 0.75, step=0.1), fontsize=20)
plt.yticks(np.arange(40, 90, step=20), fontsize=20)
plt.ylabel('Classification Accuracy', fontsize=30)
plt.xlabel('p2', fontsize=30)
plt.ylim([40, 90])
#plt.show()
plt.savefig("stability_power_p2.eps")
