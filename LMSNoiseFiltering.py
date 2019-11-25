#%%
import numpy as np
import matplotlib.pylab as plt
from padasip.filters.base_filter import AdaptiveFilter
import padasip as pa
from scipy.io import wavfile

class FilterLMS(AdaptiveFilter):

    def __init__(self, n, mu=0.1, w="random"):
        self.kind = "LMS filter"
        if type(n) == int:
            self.n = n
        else:
            raise ValueError('The size of filter must be an integer') 
        self.mu = self.check_float_param(mu, 0, 1000, "mu")
        self.init_weights(w, self.n)
        self.w_history = False   

    def run(self, d, x):
        # measure the data and check if the dimme
        N = len(x)
        if not len(d) == N:
            raise ValueError('The length of vector d and matrix x must agree.')  
        # self.n = len(x[0])
        # prepare data
        try:    
            x = np.array(x)
            d = np.array(d)
        except:
            raise ValueError('Impossible to convert x or d to a numpy array')
        # create empty arrays
        y = np.zeros(N)
        e = np.zeros(N)
        self.w_history = np.zeros((N,self.n))
        print("N = ", N)
        # adaptation loop
        for k in range(self.n, N):
            y[k] = np.dot(self.w, x[k - self.n : k])
            e[k] = d[k] - y[k]
            # self.w = self.w + self.mu * x[k - self.n : k]
            self.w = self.w + self.mu * x[k - self.n : k] * e[k]
        return y, e, self.w_history  

(sample_rate_rawdata, rawdata) = wavfile.read('/home/yingbo/FlecksCode/python/lms/PairProgrammingRawData.wav')
(sample_rate_backgroundnoise, backgroundnoise) = wavfile.read('/home/yingbo/FlecksCode/python/lms/BackgroundNoise.wav')
# select_points = 100000
select_points = rawdata.shape[0]

rawdata__0 = rawdata[0:select_points, 0]
backgroundnoise_0 = backgroundnoise[0:select_points, 0]
sample_points = rawdata__0.shape[0]
time = np.arange(0, sample_points) * (1 / sample_rate_rawdata)

# plt.figure(1)
# plt.plot(time, rawdata__0, "b")
# plt.plot(time, backgroundnoise_0, "r")
# plt.ylim(-20000, 20000)
# plt.show()

x = backgroundnoise_0 / 20000
d = rawdata__0 / 20000

# identification
f = FilterLMS(n=4, mu=0.5, w="random")
y, e, w = f.run(d, x)

wavfile.write("signal_after_noise_removal.wav", sample_rate_rawdata, e)
wavfile.write("signal.wav", sample_rate_rawdata, d)

# show results

plt.figure(1)
plt.subplot(311);plt.xlabel("iteration - k")
plt.plot(rawdata__0,"b", label="signal_mixed_with_noise")
plt.plot(backgroundnoise_0,"r", label="noise")
plt.legend()
plt.subplot(312);plt.xlabel("iteration - k")
plt.plot(e,"g", label="signal_noise_removal")
plt.legend()
plt.subplot(313);plt.xlabel("iteration - k")
plt.plot(y,"g", label="noise_approximation")
plt.legend()
# plt.subplot(414);plt.title("Filter error");plt.xlabel("samples - k")
# plt.plot(10*np.log10(e**2),"r", label="e - error [dB]");plt.legend()
# plt.tight_layout()
# plt.rcParams['agg.path.chunksize'] = 1000
plt.show()