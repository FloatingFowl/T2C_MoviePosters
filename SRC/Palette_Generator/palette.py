import time
import matplotlib.pyplot as plt
import math
import numpy as np
import cv2

class Palette:

    def __init__(self, img_fn, k=5):
        self.img = cv2.imread(img_fn)
        self.K = k
        self.bin_size = 16
        self.bin_range = 16
        self.bins = dict()
        for i in range(16):
            for j in range(16):
                for k in range(16):
                    tmp = dict()
                    tmp["color"] = [(i+0.5)*16,(j+0.5)*16,(k+0.5)*16]
                    tmp["count"] = 0
                    tmp["idx"] = -1
                    tmp["Lab"] = self.rgb2lab(tmp["color"])
                    self.bins['r'+str(i)+'g'+str(j)+'b'+str(k)] = tmp

    def rgb2lab(self, col):
        RGB = [0.0,0.0,0.0]
        for i in range(3):
            v = col[i]/255.0
            if v > 0.04045:
                v = math.pow((v+0.055)/1.055, 2.4)
            else:
                v/=12.92
            RGB[i] = 100.0 * v
        X = RGB[0] * 0.4124 + RGB[1] * 0.3576 + RGB[2] * 0.1805;
        Y = RGB[0] * 0.2126 + RGB[1] * 0.7152 + RGB[2] * 0.0722;
        Z = RGB[0] * 0.0193 + RGB[1] * 0.1192 + RGB[2] * 0.9505;
        XYZ = [X, Y, Z];
        XYZ[0] /= 95.047;
        XYZ[1] /= 100.0;
        XYZ[2] /= 108.883;
        for i in range(3):
            v = XYZ[i]
            if v > 0.008856:
                v = math.pow(v, 1/3.0)
            else:
                v *= 7.787
                v += 16 / 116.0
            XYZ[i] = v
        L = (116 * XYZ[1]) - 16;
        a = 500 * (XYZ[0] - XYZ[1]);
        b = 200 * (XYZ[1] - XYZ[2]);
        Lab = [L, a, b];
        return Lab;

    def lab2rgb(self, col):

        L = col[0];
        a = col[1];
        b = col[2];

        d = 6 / 29;
        fy = (L + 16) / 116;
        fx = fy + a / 500;
        fz = fy - b / 200;
        if fy > d:
            Y = fy ** 3
        else:
            Y = (fy - 16 / 116.0) * 3 * d * d
        if fx > d:
            X = fx ** 3
        else:
            X = (fx - 16 / 116.0) * 3 * d * d
        if fz > d:
            Z = fz ** 3
        else:
            Z = (fz - 16 / 116.0) * 3 * d * d

        X *= 95.047;
        Y *= 100.0;
        Z *= 108.883;
        R = 3.2406 * X + (-1.5372) * Y + (-0.4986) * Z
        G = (-0.9689) * X + 1.8758 * Y + 0.0415 * Z
        B = 0.0557 * X + (-0.2040) * Y + 1.0570 * Z

        RGB = [R, G, B];
        for i in range(3):
            v = RGB[i] / 100;
            if v > 0.0405 / 12.92:
                v = math.pow(v, 1 / 2.4)
                v *= 1.055;
                v -= 0.055;
            else:
                v *= 12.92;
            RGB[i] = round(v * 255);

        return RGB;

    def isOutRGB(self, RGB):
        for i in range(3):
            if RGB[i] < 0 or RGB[i] > 255:
                return True
        return False

    def isOutLab(self, Lab):
        return self.isOutRGB(self.lab2rgb(Lab))

    def isEqual(self, c1, c2):
        for i in range(len(c1)):
            if c1[i] != c2[i]:
                return False
        return True

    def labBoundary(self, pin, pout):
        mid = []
        for i in range(len(pin)):
            mid.append((pin[i] + pout[i]) / 2)
        RGBin = self.lab2rgb(pin);
        RGBout = self.lab2rgb(pout);
        RGBmid = self.lab2rgb(mid);
        if self.distance2(pin,pout)<0.001 or self.isEqual(RGBin, RGBout):
            return mid;
        if self.isOutRGB(RGBmid):
            return self.labBoundary(pin, mid);
        else:
            return self.labBoundary(mid, pout);
    
    def labIntersect(self, p1, p2):
        if self.isOutLab(p2) is True:
            return self.labBoundary(p1,p2)
        else:
            return self.labIntersect(p2, self.add(p2, self.sub(p2,p1)))

    def palette(self):
        for i in range(self.img.shape[0]):
            for j in range(self.img.shape[1]):
                R = self.img[i,j,2];
                G = self.img[i,j,1];
                B = self.img[i,j,0];
                ri = math.floor(R / 16.0)
                gi = math.floor(G / 16.0)
                bi = math.floor(B / 16.0)
                self.bins['r' + str(ri) + 'g' + str(gi) + 'b' + str(bi)]["count"]+=1


    def distance2(self, c1, c2):
        res = 0;
        for i in range(len(c1)):
            res += (c1[i] - c2[i]) * (c1[i] - c2[i]);
        return res;

    def normalize(self, v):
        d = math.sqrt(self.distance2(v, [0, 0, 0]))
        res = []
        for i in range(len(v)):
            res.append(v[i]/d)
        return res

    def add(self, c1, c2):
        res = []
        for i in range(len(c1)):
            res.append(c1[i] + c2[i])
        return res

    def sub(self, c1, c2):
        res = []
        for i in range(len(c1)):
            res.append(c1[i] - c2[i])
        return res

    def sca_mul(self, c, k):
        res = []
        for i in range(len(c)):
            res.append(c[i] * k)
        return res

    def kmeansFirst(self):
        centers_lab = [];
        centers_lab.append(self.rgb2lab([8.0,8.0,8.0]));
        bins_copy = {};
        for i in self.bins.keys():
            bins_copy[i] = self.bins[i]["count"];

        for p in range(self.K):
            tmp = None
            maxc = -1
            for i in bins_copy.keys():
                d2 = self.distance2(self.bins[i]["Lab"], centers_lab[p])
                factor = 1 - math.exp(-d2 / 6400)
                bins_copy[i] *= factor
                if bins_copy[i] > maxc:
                    maxc = bins_copy[i]
                    tmp = []
                    for j in range(3):
                        tmp.append(self.bins[i]["color"][j]);
            centers_lab.append(self.rgb2lab(tmp));
        return centers_lab;

    def kmeans(self):
        centers = self.kmeansFirst()
        no_change = False
        while no_change is False:
            no_change = True
            sum = []
            for i in range(self.K+1):
                sum.append({
                        "color": [0.0, 0.0, 0.0],
                        "count": 0
                });
            for i in range(self.bin_range):
                for j in range(self.bin_range):
                    for k in range(self.bin_range):
                        tmp = self.bins['r' + str(i) + 'g' + str(j) + 'b' + str(k)];
                        if tmp["count"] == 0:
                            continue
                        lab = tmp["Lab"];
                        mind = math.inf
                        mini = -1;
                        for p in range(self.K + 1):
                            d = self.distance2(centers[p], lab);
                            if mind > d:
                                mind = d
                                mini = p
                        if mini != tmp["idx"]:
                            tmp["idx"] = mini
                            no_change = False
                        m = self.sca_mul(tmp["Lab"], tmp["count"]);
                        sum[mini]["color"] = self.add(sum[mini]["color"], m);
                        sum[mini]["count"] += tmp["count"];
            for i in range(1, self.K+1):
                if sum[i]["count"] != 0:
                    for j in range(3):
                        centers[i][j] = sum[i]["color"][j] / sum[i]["count"];
        centers = self.sorti(centers);
        centers_rgb = [];
        for i in range(0, self.K+1):
            centers_rgb.append(self.lab2rgb(centers[i]));
        return centers_rgb;

    def sorti(self, colors):
        l = len(colors)
        for i in range(l-1,0,-1):
            for j in range(i):
                if colors[j][0] > colors[j+1][0]:
                    tmp = colors[j]
                    colors[j] = colors[j+1]
                    colors[j+1] = tmp
        return colors

    def colorTransform(self, colors1, colors2):
        colors1 = [self.rgb2lab(i) for i in colors1]
        colors2 = [self.rgb2lab(i) for i in colors2]
        self.L1 = [0];
        self.L2 = [0];
        for i in range(1,len(colors1)):
            self.L1.append(colors1[i][0])
            self.L2.append(colors2[i][0])
        self.L1.append(100)
        self.L2.append(100)

        finim = np.array(self.img,dtype=float)

        cs1 = []
        cs2 = []
        k = 0

        for i in range(1, self.K+1):
            cs1.append(colors1[i])
            cs2.append(colors2[i])
            k+=1
        self.sigma = self.getSigma(colors1)
        self.lambdaa = self.getLambda(cs1)

        ama = False
        sm = 0

        for i in range(self.img.shape[0]):
            for j in range(self.img.shape[1]):
                if i==91 and j==76:
                    ama = True
                R = self.img[i,j,2]/1.0;
                G = self.img[i,j,1]/1.0;
                B = self.img[i,j,0]/1.0;

                Lab = self.rgb2lab([R, G, B]);
                out_lab = [0.0, 0.0, 0.0];

                L = self.colorTransformSingleL(Lab[0]);
                for p in range(k):
                    v = self.colorTransformSingleAB([cs1[p][1], cs1[p][2]], [cs2[p][1], cs2[p][2]], Lab[0], Lab);
                    v[0] = L;
                    omegaw = self.omegar(cs1, Lab, p);
                    v = self.sca_mul(v, omegaw);
                    out_lab = self.add(out_lab, v);

                out_rgb = self.lab2rgb(out_lab);
                sm += out_rgb[0] + out_rgb[1] + out_rgb[2] 

                finim[i,j,0] = out_rgb[0]
                finim[i,j,1] = out_rgb[1]
                finim[i,j,2] = out_rgb[2]
                finim[i,j,0] = out_lab[0]
                finim[i,j,1] = out_lab[1]
                finim[i,j,2] = out_lab[2]
            if i%100 == 0:
                print(i)
            # if i > 400:
                # break
        finim[:,:,0] = finim[:,:,0] * 2.55
        finim[:,:,1] += 128.0
        finim[:,:,2] += 128.0
        finim[finim>255] = 255
        finim[finim<0] = 0
        finim = finim.astype(np.uint8)
        print(finim.shape)
        finim = cv2.cvtColor(finim, cv2.COLOR_Lab2RGB)
        return finim

    def colorTransformSingleL(self, l):
        i = 0
        while i < len(self.L1)-1:
            if l >= self.L1[i] and l <= self.L1[i + 1]:
                break;
            i += 1
        l1 = self.L1[i];
        l2 = self.L1[i + 1];
        if l1 == l2:
            s = 1
        else:
            s = (l-l1)/(l2-l1)
        Lo1 = self.L2[i];
        Lo2 = self.L2[i + 1];
        L = (Lo2 - Lo1) * s + Lo1;
        return L

    def colorTransformSingleAB(self, ab1, ab2, L, x):
        color1 = [L, ab1[0], ab1[1]];
        color2 = [L, ab2[0], ab2[1]];
        if self.distance2(color1, color2) < 0.0001:
            return color1;
        d = self.sub(color2, color1);
        x0 = self.add(x, d);
        Cb = self.labIntersect(color1, color2);
        if self.isOutLab(x0):
            xb = self.labBoundary(color2, x0);
        else:
            xb = self.labIntersect(x, x0);
        dxx = self.distance2(xb, x);
        dcc = self.distance2(Cb, color1);
        l2 = min(1, dxx / dcc);
        xbn = self.normalize(self.sub(xb, x));
        x_x = math.sqrt(self.distance2(color1, color2) * l2);
        return self.add(x, self.sca_mul(xbn, x_x));

    def omegar(self, cs1, Lab, i):
        sum = 0;
        for j in range(len(cs1)):
            sum += self.lambdaa[j][i] * self.phi(math.sqrt(self.distance2(cs1[j], Lab)));
        return sum;

    def getLambda(self, cs1):
        s = [];
        k = len(cs1);
        for p in range(k):
            tmp = [];
            for q in range(k):
                tmp.append(self.phi(math.sqrt(self.distance2(cs1[p], cs1[q]))));
            s.append(tmp);
        lambdaa = np.linalg.inv(s).tolist();
        return lambdaa

    def phi(self, r):
        return math.exp(-r * r / (2 * self.sigma * self.sigma));

    def getSigma(self, colors):
        sum = 0;
        for i in range(self.K+1):
            for j in range(self.K+1):
                if i == j:
                    continue
                sum += math.sqrt(self.distance2(colors[i], colors[j]));
        return sum / (self.K * (self.K + 1));

'''
a = Palette('1499_biography.txt.jpg')
a.palette()
z = a.kmeans()
print(z)
b = np.zeros((100,500,3), dtype=np.uint8)
for i in range(5):
     b[:,i*100:(i+1)*100,:] = np.array(z[i+1])
# plt.imshow(b)
# plt.show()

temp = [ [0,0,0],
[22.4590175152944, -1.10785867891941, 3.142493867304841],
[47.2956491777223, 19.393337882207206, 20.91461775182927],
[61.600325220852724, 3.0763166163803035, 15.857161769470963],
[81.34204131669767, 2.11613072292266, 11.38349886776886],
[94.91777419157178, 4.734314621225922, -7.585055824397036],
    ]

newim = a.colorTransform(z, temp)
'''
