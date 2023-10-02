"""
Class Module / Loss.
"""

class Loss(object):
    """ Classe abstraite pour le calcul du coût.
        Note: y et yhat sont des matrices de taille batch × d : chaque
             supervision peut être un vecteur de taille d, pas seulement un
             scalaire comme dans le cas de la régression univariée.
    """
    def forward(self, y, yhat):
        """Calculer le cout en fonction de deux entrees"""
        pass

    def backward(self, y, yhat):
        """calcul le gradient du cout par rapport yhat"""
        pass

class Module(object):
    """ Classe abstraite représentant un module générique du réseau de
        neurones. Ses attributs sont les suivants:
            * self._parameters: obj, stocke les paramètres du module, lorsqu'il
            y en a (ex: matrice de poids pour un module linéaire)
            * self._gradient: obj, permet d'accumuler le gradient calculé
    """
    def __init__(self):
        """Initialisation des parametres"""
        self._parameters = None
        self._gradient = None

    def zero_grad(self):
        """Annule gradient"""
        pass


    def forward(self, X):
        """Calcule la passe forward"""
        pass

    def update_parameters(self, gradient_step=1e-3):
        """Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step"""
        self._parameters -= gradient_step*self._gradient

    def backward_update_gradient(self, input, delta):
        """Met a jour la valeur du gradient"""
        pass

    def backward_delta(self, input, delta):
        """Calcul la derivee de l'erreur"""
        pass