"""
Class Autoencodeur.
"""
import numpy as np

from loss import BCELoss
from Linear import Linear
from activation import TanH, Sigmoide


class AutoEncodeur :
    """ Classe pour l'auto-encodage
    """
    def codage (self, xtrain, modules):
        """ Encodage.
        """
        res_forward = [ modules[0].forward(xtrain) ]

        for j in range(1, len(modules)):
            res_forward.append( modules[j].forward( res_forward[-1] ) )

        return res_forward

    def fit(self, xtrain, ytrain, batch_size=1, neuron=10, niter=1000, gradient_step=1e-5):
        """ Classifieur non-linéaire sur les données d'apprentissage.
        """
        # Ajout d'un biais aux données
        # xtrain = add_bias (xtrain)

        # Récupération des tailles des entrées
        batch, output = ytrain.shape
        batch, input = xtrain.shape

        # Initialisation des couches du réseau et de la loss
        self.bce = BCELoss()
        self.linear_1 = Linear(input, neuron)
        self.tanh = TanH()
        self.linear_2 = Linear(neuron, output)
        self.sigmoide = Sigmoide()
        self.linear_3 = Linear (output, neuron)
        self.linear_4 = Linear (neuron, input)


        # Liste des couches du réseau de neurones
        self.modules_enco = [ self.linear_1, self.tanh, self.linear_2, self.tanh ]
        self.modules_deco = [ self.linear_3, self.tanh, self.linear_4, self.sigmoide ]
        self.net = self.modules_enco + self.modules_deco

        for i in range(niter):
            res_forward_enco = self.codage(xtrain, self.modules_enco)
            res_forward_deco = self.codage(res_forward_enco[-1], self.modules_deco)
            res_forward = res_forward_enco + res_forward_deco
            if(i%100==0):
                print(np.sum(np.mean(self.bce.forward(xtrain, res_forward[-1]), axis=1)))

            # Phase backward 
            deltas =  [ self.bce.backward( xtrain, res_forward[-1] ) ]

            for j in range(len(self.net) - 1, 0, -1):
                deltas += [self.net[j].backward_delta( res_forward[j-1], deltas[-1] ) ]

            #Phase backward et mise-à-jour
            for j in range(len(self.net)):
                # Mise-à-jour du gradient
                if j == 0:
                    self.net[j].backward_update_gradient(xtrain, deltas[-1])
                else:
                    self.net[j].backward_update_gradient(res_forward[j-1], deltas[-j-1])

                # Mise-à-jour des paramètres
                self.net[j].update_parameters(gradient_step)
                self.net[j].zero_grad()

    def predict (self, xtest) :
        res_forward_enco = self.codage(xtest, self.modules_enco)
        res_forward_deco = self.codage(res_forward_enco[-1], self.modules_deco)
        return res_forward_enco[-1], res_forward_deco[-1]