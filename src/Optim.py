"""
Class Optim avec SGD (Stochastic Gradient Descent)
"""
import numpy as np

from tqdm import tqdm

class Optim :
    """ Classe qui permet de condenser une itération de gradient. Elle calcule
        la sortie du réseau self.net, exécute la passe backward et met à jour
        les paramètres du réseau.
            * self.net: list(Module), réseau de neurones sous forme d'une liste
                        de Modules correspondant aux différentes couches.
            * self.loss: Loss, coût à minimiser
            * self.eps: float, pas pour la mise-à-jour du gradient
    """
    def __init__(self,net, loss, eps=1e-3):
        """Initialisation des parametres"""
        self.net=net
        self.loss=loss
        self.eps=eps

    def step(self,batch_x,batch_y):
        pass_forward=self.net.forward(batch_x)
        loss = self.loss.forward(batch_y,pass_forward).mean()
        backward_loss=self.loss.backward(batch_y,pass_forward)
        self.net.backward_delta(batch_x,backward_loss)
        self.net.backward_update_gradient(batch_x,backward_loss)
        self.net.update_parameters(self.eps)
        self.net.zero_grad()
        return loss

    def update(self):
        """methode abstraite à enrichir dans un optimizer"""
        pass

class SGD(Optim):
    """Appliquer une descente de gradient stochastique
    """
    
    def __init__(self, net, loss,datax,datay,batch_size=20,nbIter=100, eps=1e-3):
        """Initialisation des parametres"""
        self.net=net
        self.loss=loss
        self.eps=eps
        self.datax=datax
        self.datay=datay
        self.batch_size=batch_size
        self.nbIter=nbIter

    def create_minibatch(self):
        """ Fonction qui renvoit un mini batch depuis les données"""
        size=len(self.datax)
        values = np.arange(size)
        np.random.shuffle(values)
        nb_batch = size // self.batch_size
        if (size % self.batch_size != 0):
            nb_batch += 1
        for i in range(nb_batch):
            index=values[i * self.batch_size:(i + 1) * self.batch_size]
            yield self.datax[index],self.datay[index]

    def update(self):
        """ Mise a jour avec descente de gradient stochastique"""
        list_loss = []
        for _ in tqdm(range(self.nbIter)):
            list_loss_batch = []
            for batch_x,batch_y in self.create_minibatch():
                list_loss_batch.append( self.step(batch_x,batch_y) )
            list_loss.append(np.mean(list_loss_batch))
        return list_loss