import numpy as np

import pygame,random,time,pickle,os
from math import sqrt
pygame.init()
timer = pygame.time.Clock()
screen = pygame.display.set_mode((1080,720))
pygame.display.set_caption("MemoryNeurones")

class TextDisplayer:
    def __init__(self):
        self.polices = {}
    def draw_text(self, surface,text,color,pos,taille):
        if self.polices.get(str(taille)+text):
            screen.blit(self.polices[str(taille)+text][0].render(text, 10, color), (pos[0]-self.polices[str(taille)+text][1][2]/2,pos[1]-self.polices[str(taille)+text][1][3]/2))
        else:
            posa = pygame.font.SysFont(None, taille).render(text, 10, color).get_rect()
            self.polices[str(taille)+text] = [pygame.font.SysFont(None, taille),[i for i in posa]]
            screen.blit(pygame.font.SysFont(None, taille).render(text, 10, color), (pos[0]-posa[2]/2,pos[1]-posa[3]/2))

class Graphic_Neural_Network(object):
    def __init__(self, zipping):
        self.zipping = zipping
        self.neuron_radius = [20,23]
        self.connection_thickness = [1,10]
    def display(self, screen, network, size, textDisplayer):
        if network.learning_rate/network.max_learning_rate < 1:
            pygame.draw.rect(screen,[0,25,50],[size[0],size[1],size[2]/2*(1-network.learning_rate/network.max_learning_rate),size[3]])
            pygame.draw.rect(screen,[0,25,50],[size[0]+size[2]/2+size[2]/2*(network.learning_rate/network.max_learning_rate),size[1],size[2]/2*(1-network.learning_rate/network.max_learning_rate),size[3]])
            pygame.draw.rect(screen,[0,25,50],[size[0],size[1],size[2],size[3]/2*(1-network.learning_rate/network.max_learning_rate)])
            pygame.draw.rect(screen,[0,25,50],[size[0],size[1]+size[3]/2+size[3]/2*(network.learning_rate/network.max_learning_rate),size[2],size[3]/2*(1-network.learning_rate/network.max_learning_rate)])
        neurons = [[i for i in [i for i in network.X[0]]]]+[[i[0] for i in network.z]][0]
        layers = [network.sigmoid(i) for i in network.w]
        neurons, layers = self.ZIPNetwork(neurons, layers, self.zipping)
        size_x, size_y, size_z, size_w = size
        layer_width = (size_z-self.neuron_radius[1]*2)/(len(neurons)-1)
        for l, layer in enumerate(layers):
            layer_len = len(layer)
            for a, layer_neurons in enumerate(layer):
                layer_neurons_len = len(layer_neurons)
                for y, value in enumerate(layer_neurons):
                    if layer_len > 1:
                        line_start = (size_x + self.neuron_radius[1] + layer_width*l, size_y + self.neuron_radius[1] + ((size_w-self.neuron_radius[1]*2)/(layer_len-1))*a)
                    else:
                        line_start = (size_x + self.neuron_radius[1] + layer_width*l, size_y + self.neuron_radius[1] + ((size_w-self.neuron_radius[1]*2)/2))
                    if layer_neurons_len > 1:
                        line_end = (size_x + self.neuron_radius[1] + layer_width*(l+1), size_y + self.neuron_radius[1] + ((size_w-self.neuron_radius[1]*2)/(layer_neurons_len-1))*y)
                    else:
                        line_end = (size_x + self.neuron_radius[1] + layer_width*(l+1), size_y + self.neuron_radius[1] + ((size_w-self.neuron_radius[1]*2)/2))
                    if value <= 0:
                        color = [(1+value)*100,100,250]
                        thickness = self.connection_thickness[0]
                    else:
                        color = [100+value*150,100,100+(1-value)*150]
                        thickness = int(value*(self.connection_thickness[1] - self.connection_thickness[0]) + self.connection_thickness[0])
                    pygame.draw.line(screen, color, line_start, line_end, thickness)
        for count, layer in enumerate(neurons):
            layer_len = len(layer)
            for i, neuron in enumerate(layer):
                if layer_len > 1:
                    circle_center = (size_x + self.neuron_radius[1] + layer_width*count, size_y + self.neuron_radius[1] + ((size_w-self.neuron_radius[1]*2)/(layer_len-1))*i)
                else:
                    circle_center = (size_x + self.neuron_radius[1] + layer_width*count, size_y + self.neuron_radius[1] + ((size_w-self.neuron_radius[1]*2)/2))
                circle_radius = int(neuron*(self.neuron_radius[1]-self.neuron_radius[0])+self.neuron_radius[0])
                color = [100+neuron*150, 100, 100+(1-neuron)*150]
                pygame.draw.circle(screen, color, circle_center, circle_radius)
                textDisplayer.draw_text(screen, str(round(neuron,2)), [color[0]*0.5,color[1]*0.5,color[2]*0.5], circle_center, circle_radius)

    def ZIPNetwork(self, neurons, layers, zipping):
        def zip_values(values, zipping):
            if len(values) <= zipping:
                return values
            return [sum(chunk) / len(chunk) for chunk in (values[i * len(values) // zipping : (i + 1) * len(values) // zipping] for i in range(zipping))]
        def zip_layer(layer, zipping):
            if len(layer) <= zipping:
                return layer
            return [[sum(values) / len(values) for values in zip(*chunk)] for chunk in (layer[i * len(layer) // zipping : (i + 1) * len(layer) // zipping] for i in range(zipping))]

        ZIP_neurons = [zip_values(layer, zipping) for layer in neurons]
        ZIP_layers = [zip_layer([zip_values(neuron, zipping) for neuron in layer], zipping) for layer in layers]
        return ZIP_neurons, ZIP_layers



class Neural_Network(object):
    def __init__(self,inpSize,hiddenSizes,outSize,max_learning_rate=1,learning_rate=0.01,learning_rate_speed=1.1):
        #Paramètres
        self.inpSize = inpSize
        self.outSize = outSize
        self.hiddenSizes = hiddenSizes
        self.default_learning_rate = learning_rate
        self.learning_rate_speed = learning_rate_speed
        self.max_learning_rate = max_learning_rate
        self.learning_rate = self.default_learning_rate
        #Génération des poids
        self.w = []
        self.b = []
        if self.inpSize > 0:
            self.w.append(np.random.randn(self.inpSize,self.hiddenSizes[0]))
            self.b.append(np.random.randn(1,self.hiddenSizes[0]))
        for i in range(len(self.hiddenSizes)-1):
            self.w.append(np.random.randn(self.hiddenSizes[i],self.hiddenSizes[1+i]))
            self.b.append(np.random.randn(1,self.hiddenSizes[1+i]))
        if self.outSize > 0:
            self.w.append(np.random.randn(self.hiddenSizes[-1],self.outSize))
            self.b.append(np.random.randn(1,self.outSize))
    # Fonctions de modification du réseau
    def addInput(self):
        if self.inpSize == 0:
            self.inpSize += 1
            self.w.insert(0,np.random.randn(self.inpSize,self.hiddenSizes[0]))
            self.b.insert(0,np.random.randn(1,self.hiddenSizes[0]))
        else:
            self.inpSize += 1
            new_weights = np.random.randn(self.inpSize,self.hiddenSizes[0])
            new_weights[:-1] = self.w[0]
            self.w[0] = new_weights
            self.b[0] = np.random.randn(1,self.hiddenSizes[0])

    def addOutput(self):
        if self.outSize == 0:
            self.outSize += 1
            self.w.append(np.random.randn(self.hiddenSizes[-1],self.outSize))
            self.b.append(np.random.randn(1,self.outSize))
        else:
            self.outSize += 1
            new_weights = np.random.randn(self.hiddenSizes[-1],self.outSize)
            new_weights[:,:-1] = self.w[-1]
            self.w[-1] = new_weights
            self.b[-1] = np.random.randn(1,self.outSize)

    # Fonction de sauvegarde et de chargement du cerveau
    def save(self, filename):
        with open(filename,'wb') as f:
            pickle.dump([self.w, self.b], f)

    def load(self, filename):
        with open(filename,'rb') as f:
            self.w, self.b = pickle.load(f)

    # Fonction de propagation avant
    def forward(self,X):
        self.X = X
        self.z = []
        self.z.append(self.sigmoid(np.dot(X,self.w[0]) + self.b[0])) # Ajout du biais
        for i in range(len(self.w)-1):
            self.z.append(self.sigmoid(np.dot(self.z[-1],self.w[1+i]) + self.b[1+i])) # Ajout du biais
        return self.z[-1]
    # Fonction d'activation
    def sigmoid(self, s):
        return 1/(1+np.exp(-s))
    # Dérivée de la fonction d'activation
    def sigmoidPrime(self, s):
        return self.sigmoid(s) * (1 - self.sigmoid(s))
    #Fonction de rétropropagation
    def backward(self, X, Y):
        # Calcul des modifications des poifs et des biais
        self.e = [0] * len(self.w)
        self.delta_b = [0] * len(self.b) # Ajout des deltas de biais
        self.e[-1] = (Y - self.z[-1]) * self.sigmoidPrime(self.z[-1])  # Erreur de la couche de sortie
        self.delta_b[-1] = np.sum(self.e[-1], axis=0) # Calcul du delta de biais pour la couche de sortie
        for i in reversed(range(len(self.w) - 1)):
            self.e[i] = self.e[i + 1].dot(self.w[i + 1].T) * self.sigmoidPrime(self.z[i])
            self.delta_b[i] = np.sum(self.e[i], axis=0) # Calcul des deltas de biais pour les couches cachées
        # Mise à jour des poids et des biais
        for i in range(len(self.w)):
            if i == 0:
                self.w[i] += self.learning_rate * X.T.dot(self.e[i])
                self.b[i] += self.learning_rate * self.delta_b[i]
            else:
                self.w[i] += self.learning_rate * self.z[i - 1].T.dot(self.e[i])
                self.b[i] += self.learning_rate * self.delta_b[i]
    def train(self,X,Y):
        o = self.forward(X)
        self.backward(X,Y)
        if self.learning_rate < self.max_learning_rate:
            self.learning_rate *= self.learning_rate_speed
            if self.learning_rate > self.max_learning_rate:
                self.learning_rate = self.max_learning_rate
        return o

# Personnalisation complete du modèle
lenInputs = 4
lenOutputs = 4
lensHiddens = [4,8,8,4]

NN = Neural_Network(lenInputs,lensHiddens,lenOutputs,max_learning_rate=1,learning_rate=0.01,learning_rate_speed=1.1)
NNGraphic = Graphic_Neural_Network(20)
textDisplayer = TextDisplayer()

X = np.array([[1,0,0,0],[0,0,0,1]])
Y = np.array([[0,0,0,1],[1,0,0,0]])


while True:
    NN.train(np.array(X),np.array(Y))
    timer.tick(30)
    screen.fill([0,50,100])
    NNGraphic.display(screen,NN,[0,0,1080,720],textDisplayer)
    pygame.display.update()
    for event in pygame.event.get():
        if event.type == 256:
            pygame.quit()
            exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                NN.learning_rate = NN.default_learning_rate
                NN.w = []
                NN.b = []
                NN.w.append(np.random.randn(NN.inpSize,NN.hiddenSizes[0]))
                NN.b.append(np.random.randn(1,NN.hiddenSizes[0]))
                for i in range(len(NN.hiddenSizes)-1):
                    NN.w.append(np.random.randn(NN.hiddenSizes[i],NN.hiddenSizes[1+i]))
                    NN.b.append(np.random.randn(1,NN.hiddenSizes[1+i]))
                NN.w.append(np.random.randn(NN.hiddenSizes[-1],NN.outSize))
                NN.b.append(np.random.randn(1,NN.outSize))
            if event.key == pygame.K_p:
                print(np.round(NN.forward(X), 3))
