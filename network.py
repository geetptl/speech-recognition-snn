import numpy as np
from matplotlib import pyplot as plt

class Izhikevich:
    def __init__(self, a, b, c, d, Vth, T, dt):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.Vth = Vth
        self.u = self.b * self.c
        self.T = T
        self.dt = dt
        self.t = np.arange(0, self.T, self.dt)
        self.in_synapes = []
        self.out_synapes = []

    # I is an array of length self.t
    def run(self, I):
        V = np.zeros(len(self.t))
        V[0] = self.c
        u = np.zeros(len(self.t))
        u[0] = self.u
        num_spikes = 0
        for t in range(1, len(self.t)):
            dv = ((0.04 * V[t - 1] ** 2) + (5 * V[t - 1]) + 140 - u[t - 1] + I[t - 1]) * self.dt
            du = (self.a * ((self.b * V[t - 1]) - u[t - 1])) * self.dt
            V[t] = V[t - 1] + dv
            u[t] = u[t - 1] + du

            if V[t] >= self.Vth:
                V[t] = self.c
                u[t] = self.d + u[t]
                num_spikes += 1

        return V, num_spikes

T = 1000
dt = 0.1
t = np.arange(0, T, dt)
izhi = Izhikevich(0.02, 0.2, -65, 2, 30, T, dt)

I = 25
i_inj = np.linspace(0, I, t.shape[0])
# i_inj = np.full(t.shape, I)
V, num_spikes = izhi.run(i_inj)

print(num_spikes)

plt.figure()
plt.subplot(2,1,1)
plt.title('Izhi Neuron')
plt.plot(t, V, 'k')
plt.ylabel('V')

plt.subplot(2,1,2)
plt.plot(t, i_inj, 'k')
plt.xlabel('t')
plt.ylabel('$I_{inj}$ ($\\mu{A}/cm^2$)')

plt.show()


class Synapse:
    def __init__(self):
        self.weight = 1

    def hebbian(self):
        pass

    def anti_hebbian(self):
        pass


class Network:
    def __init__(self):
        self.layer_1 = [Izhikevich() for i in range(200)]
        self.synapse_layer = [[Synapse() for j in range(200)] for i in range(10)]
        self.layer_2 = [Izhikevich() for i in range(10)]

        for i, synapse1_ in enumerate(self.synapses):
            for j, synapse_ in enumerate(synapse1_):
                layer_2[i].in_synapes.append(synapse_)
                layer_1[j].out_synapes.append(synapse_)

    def forward(self, feature, label):
        # print("hello")
        pass
