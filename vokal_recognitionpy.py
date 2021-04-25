import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy import signal
import sounddevice as sd
import time
import pysptk 



people = ['m', 'w', 'b', 'g']
classes = ['ae', 'ah', 'aw', 'eh', 'ei', 'er', 'ih', 'iy','oa', 'oo', 'uh', 'uw']
avslag = ['Nei', 'nei', 'No', 'Nope', 'no', 'nope', 'n']
C = 12
Fs = 16000
GMM_number = 3


mu =  [np.array([[ 591.49349236, 1912.50992355, 2591.22584184],
       [ 624.2       , 2541.        ,    0.        ],
       [ 697.1793304 , 2400.10853874, 3107.23046549]]), np.array([[ 963.97358086, 1612.0936153 , 2863.39038548],
       [1069.66666667,    0.        , 2923.        ],
       [ 766.62305769, 1341.10015312, 2615.95765414]]), np.array([[ 950.        ,    0.        , 2955.        ],
       [ 635.        , 1163.        ,    0.        ],
       [ 763.80740741, 1170.85925926, 2762.6       ]]), np.array([[ 705.14898422, 2116.68165268, 2998.0314528 ],
       [ 581.6684733 , 1793.84804368, 2590.0328176 ],
       [ 807.5191198 , 2285.3407006 , 3442.36534737]]), np.array([[ 548.38263345, 2578.24992699, 3184.38602757],
       [ 526.85714286, 2648.71428571,    0.        ],
       [ 481.06997669, 2083.45613523, 2694.80857687]]), np.array([[ 496.8       , 1553.86666667,    0.        ],
       [ 483.9431512 , 1462.6526521 , 1800.33508871],
       [ 590.7074655 , 1689.05260762, 2104.30900481]]), np.array([[ 478.96627823, 2353.89386487, 3020.4665573 ],
       [ 513.09307592, 2565.21247738, 3437.53969635],
       [ 432.10994604, 2020.81943858, 2667.435509  ]]), np.array([[ 374.43172946, 2456.68709207, 3092.60836627],
       [ 441.16666667, 2957.66666667,    0.        ],
       [ 448.19566491, 2986.65949245, 3641.25356398]]), np.array([[ 546.62341621, 1000.70430885, 2690.09535887],
       [ 603.11245855,  373.98225894, 2641.74624817],
       [ 562.481108  , 1149.61783498, 3097.66771732]]), np.array([[ 511.8478501 , 1239.75662051, 2889.69393915],
       [ 586.28780308, 1545.00995072, 3081.50249188],
       [ 476.07609921, 1124.84492869, 2493.52916447]]), np.array([[ 713.36135877, 1336.1527646 , 2780.35643643],
       [ 709.76985346, 1553.70149569, 3122.23158743],
       [ 632.23925517,  976.05738302, 2720.82617118]]), np.array([[ 379.0520588 ,  982.42051405, 2363.53226861],
       [ 490.9098739 , 1534.3602661 , 2981.25959873],
       [ 469.41403641, 1107.7490993 , 2754.35242883]])]


sigma =  [np.array([[[ 1.71578555e+03, -1.17350318e+03, -1.10269586e+03],
        [-1.17350318e+03,  1.11798147e+04,  7.84279371e+03],
        [-1.10269586e+03,  7.84279371e+03,  1.32390614e+04]],

       [[ 5.86776000e+03, -3.69980000e+03,  0.00000000e+00],
        [-3.69980000e+03,  1.04168000e+04,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  1.00000000e-06]],

       [[ 6.66029558e+03, -2.13585632e+02,  4.86829532e+03],
        [-2.13585632e+02,  3.23305944e+04,  3.89569907e+04],
        [ 4.86829532e+03,  3.89569907e+04,  7.93043961e+04]]]), np.array([[[1.27874414e+04, 6.75506542e+03, 9.81062940e+03],
        [6.75506542e+03, 2.93753351e+04, 2.18410232e+04],
        [9.81062940e+03, 2.18410232e+04, 5.74561645e+04]],

       [[2.27902222e+04, 0.00000000e+00, 8.47466667e+03],
        [0.00000000e+00, 1.00000000e-06, 0.00000000e+00],
        [8.47466667e+03, 0.00000000e+00, 6.19940000e+04]],

       [[3.76783321e+03, 3.11006445e+03, 6.70636489e+03],
        [3.11006445e+03, 1.57157837e+04, 2.61371053e+03],
        [6.70636489e+03, 2.61371053e+03, 4.53717105e+04]]]), np.array([[[8.91800000e+03, 0.00000000e+00, 2.60680000e+04],
        [0.00000000e+00, 1.00000000e-06, 0.00000000e+00],
        [2.60680000e+04, 0.00000000e+00, 8.03086667e+04]],

       [[1.00000000e-06, 3.41212003e-24, 0.00000000e+00],
        [3.41212003e-24, 1.00000000e-06, 0.00000000e+00],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e-06]],

       [[1.11852666e+04, 1.39986618e+04, 1.67815081e+04],
        [1.39986618e+04, 2.83790395e+04, 2.30101289e+04],
        [1.67815081e+04, 2.30101289e+04, 7.89310844e+04]]]), np.array([[[ 4330.31264138, -2933.80227284, -1551.05268062],
        [-2933.80227284, 33960.31158972, 23129.78682287],
        [-1551.05268062, 23129.78682287, 39235.18735666]],

       [[  958.2700418 ,   640.47297142,  1053.36417373],
        [  640.47297142,  8529.44456428,  5158.46958355],
        [ 1053.36417373,  5158.46958355, 13782.40832027]],

       [[ 8750.62728906,  3384.26935016,  1890.87584391],
        [ 3384.26935016, 24705.29679433, 14178.58389369],
        [ 1890.87584391, 14178.58389369, 20135.08181487]]]), np.array([[[ 4.14870120e+03, -1.10908710e+02,  2.58637908e+03],
        [-1.10908710e+02,  2.32179306e+04,  2.67643706e+04],
        [ 2.58637908e+03,  2.67643706e+04,  6.38027338e+04]],

       [[ 8.70669388e+03,  1.68571020e+04,  0.00000000e+00],
        [ 1.68571020e+04,  1.11098204e+05,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  1.00000000e-06]],

       [[ 1.15574513e+03, -1.02499934e+03,  9.59208469e+02],
        [-1.02499934e+03,  1.52986274e+04,  1.10222643e+04],
        [ 9.59208469e+02,  1.10222643e+04,  2.38529713e+04]]]), np.array([[[1.22496000e+03, 5.01640000e+02, 0.00000000e+00],
        [5.01640000e+02, 2.61147822e+04, 0.00000000e+00],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e-06]],

       [[7.98239947e+02, 1.60575276e+03, 2.36221169e+03],
        [1.60575276e+03, 1.86437211e+04, 1.52715198e+04],
        [2.36221169e+03, 1.52715198e+04, 2.17513952e+04]],

       [[2.05263735e+03, 3.04792969e+02, 2.52678022e+03],
        [3.04792969e+02, 2.20748894e+04, 2.32231566e+04],
        [2.52678022e+03, 2.32231566e+04, 4.44530652e+04]]]), np.array([[[ 1.53725767e+03, -6.73823898e+02, -3.84864447e+00],
        [-6.73823898e+02,  2.09714226e+04,  1.10376072e+04],
        [-3.84864447e+00,  1.10376072e+04,  1.47132403e+04]],

       [[ 1.41208731e+03, -9.76946653e+02,  1.94470675e+03],
        [-9.76946653e+02,  2.20246127e+04,  1.04716517e+04],
        [ 1.94470675e+03,  1.04716517e+04,  2.87723156e+04]],

       [[ 8.93764488e+02,  3.64355976e+02,  1.02738933e+03],
        [ 3.64355976e+02,  8.63314171e+03,  3.88389672e+03],
        [ 1.02738933e+03,  3.88389672e+03,  8.96035799e+03]]]), np.array([[[2.91447988e+03, 8.89922099e+03, 5.34961856e+03],
        [8.89922099e+03, 5.25825019e+04, 2.87321152e+04],
        [5.34961856e+03, 2.87321152e+04, 5.20069649e+04]],

       [[1.57047222e+03, 3.88322222e+03, 0.00000000e+00],
        [3.88322222e+03, 7.20100556e+04, 0.00000000e+00],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e-06]],

       [[2.09386665e+03, 7.79028994e+02, 1.28743945e+03],
        [7.79028994e+02, 4.50524123e+04, 3.48349093e+04],
        [1.28743945e+03, 3.48349093e+04, 7.39619233e+04]]]), np.array([[[  5519.23899179,   7683.02550953,  10993.98106071],
        [  7683.02550953,  16957.39356566,  17878.1789828 ],
        [ 10993.98106071,  17878.1789828 ,  65006.31758315]],

       [[  6247.6701149 , -29037.7625754 ,  12945.09785632],
        [-29037.7625754 , 175114.09283479, -63273.03051344],
        [ 12945.09785632, -63273.03051344,  27062.55374311]],

       [[  6506.82470648,  13233.10981731,   1307.32410564],
        [ 13233.10981731,  37360.29214959,  -4138.92225431],
        [  1307.32410564,  -4138.92225431,  25383.85300847]]]), np.array([[[ 1.13723679e+03,  2.54386802e+02,  3.58921784e+02],
        [ 2.54386802e+02,  1.76711084e+04, -6.31352675e+02],
        [ 3.58921784e+02, -6.31352675e+02,  2.53861268e+04]],

       [[ 1.96153454e+03,  5.15342663e+01,  2.77208070e+02],
        [ 5.15342663e+01,  1.95771649e+04,  5.33820480e+03],
        [ 2.77208070e+02,  5.33820480e+03,  5.20162678e+04]],

       [[ 8.25548318e+02,  5.61140812e+02,  1.45284318e+03],
        [ 5.61140812e+02,  7.70925581e+03, -6.54115542e+02],
        [ 1.45284318e+03, -6.54115542e+02,  2.63291774e+04]]]), np.array([[[   9329.63043804,   12056.71749279,   21241.83210559],
        [  12056.71749279,   24811.39969442,   31088.84942766],
        [  21241.83210559,   31088.84942766,   73997.57242723]],

       [[   3149.81967455,    2455.50609417,    3324.43968185],
        [   2455.50609417,   17107.44823439,    6241.90226669],
        [   3324.43968185,    6241.90226669,   30036.48284129]],

       [[    184.109271  ,   -3592.09946463,    2790.4613278 ],
        [  -3592.09946463,  146080.31323881, -111595.51392397],
        [   2790.4613278 , -111595.51392397,   86372.58932112]]]), np.array([[[  1062.03278302,   1039.60646605,   -323.71420675],
        [  1039.60646605,   8200.51092494,  -7218.79681607],
        [  -323.71420675,  -7218.79681607, 181832.24024718]],

       [[  2578.85449378,   3399.58429965,  -2905.86536366],
        [  3399.58429965, 101117.15308826,  17086.96228468],
        [ -2905.86536366,  17086.96228468,  36170.58048122]],

       [[  1493.04288264,  -1389.4011042 ,   5200.85014983],
        [ -1389.4011042 ,  59428.59553534,   2438.17739549],
        [  5200.85014983,   2438.17739549,  51151.56459485]]])]

c =  [np.array([0.30299046, 0.03597122, 0.66103832]), np.array([0.59836797, 0.02158273, 0.3800493 ]), np.array([0.02158273, 0.00719424, 0.97122302]), np.array([0.53010816, 0.29681583, 0.173076  ]), np.array([0.63376144, 0.05035971, 0.31587885]), np.array([0.10791367, 0.48575534, 0.40633099]), np.array([0.3447646 , 0.34337512, 0.31186028]), np.array([0.48832128, 0.08633094, 0.42534778]), np.array([0.78104841, 0.02597984, 0.19297175]), np.array([0.27265903, 0.30899896, 0.41834201]), np.array([0.69816084, 0.24564924, 0.05618992]), np.array([0.32399005, 0.21180813, 0.46420182])]

def find_class_index(name):
    vowel = name[3:5]
    index = classes.index(vowel)
    return index


def find_person_index(name):
    person = name[0]
    index = people.index(person)
    return index


def gauss(xk, sigma0, mu0):
    det_sigma0 = np.linalg.det(sigma0)
    inv_sigma0 = np.linalg.inv(sigma0)
    #print(inv_sigma0)
    diff = xk - mu0
    #print(xk, mu0, diff)
    exp1 = np.dot(diff.T, inv_sigma0)
    #print(exp1)
    eksponent = (-1/2)*np.dot(exp1, diff)
    #print(eksponent)
    p = np.exp(eksponent)/np.sqrt(((2*np.pi)**3)*det_sigma0)
    #print(p, '\n')
    return p

    
def find_class(xk):
    global mu
    global sigma
    global c
    global C
    xk = np.array(xk)
    prob =[]
    for i in range(C):
        mui = mu[i]
        ci = c[i]
        sigmai = sigma[i]     
        p = 0
        for i in range(GMM_number):
            mu0 = np.array(mui[i])
            sigma0 = np.array(sigmai[i])
            p += gauss(xk, sigma0, mu0)*ci[i]
        prob.append(p)
    return np.argmax(prob)


def find_dig_cov(vec):
    ide = np.identity(3)
    return np.multiply(vec, ide)

def normalize(vec):
    vec_new = []
    for i in range(len(vec)):
        vec_new.append(vec[i]/np.max(vec))
    return np.array(vec_new)

def opptak(duration):
    global Fs
    print('Spiller inn lyd..')
    rec = sd.rec(int(duration*Fs),samplerate=Fs, channels=2)
    time.sleep(duration/4)
    print('.')
    time.sleep(duration/4)
    print('.')
    time.sleep(duration/4)
    print('.')
    time.sleep(duration/4)
    print('Prosessere...')
    rec_ny = []
    for i in range(len(rec)):
        rec_ny.append(rec[i][0])
    return normalize(rec_ny)

def fourier(vector):
    sp = np.fft.fft(vector)
    freq = np.fft.fftfreq(vector.shape[-1])
    return abs(sp.real[0:int(len(sp)/2)]), freq[0:int(len(sp)/2)]

def find_peaks(vec):
    ak = pysptk.sptk.lpc((vec), order=16)
    ak[0] = 1
    w, h = signal.freqz([1], ak,  fs = Fs)
    h = normalize(abs(h.real))
    peaks = signal.find_peaks(h, height = 0.0065, distance = 20 )    
    freq_peaks = list(w[peaks[0]])
    if len(freq_peaks)==3:
        return freq_peaks
    elif len(freq_peaks)>3:
        return freq_peaks[0:3]
    else:
        while(len(freq_peaks)<3):
            freq_peaks.append(0)
        return freq_peaks

if __name__ == "__main__":
    true = input('Trykk hvasomhelst for å komme igang')
    while(true not in avslag):
        rec = opptak(1)
        xk = find_peaks(np.array(rec))
        class_rec = find_class(xk)
        print('Vokal detektert:', classes[class_rec])
        true = input('Vil du prøve igjen?')

        
    