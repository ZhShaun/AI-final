import numpy as np

class_list = {'airplane': 0, 'bee': 1, 'bicycle': 2, 'bird': 3, 'butterfly': 4, 'cake': 5,
    'camera': 6, 'cat': 7, 'chair': 8, 'clock': 9, 'computer': 10,
    'diamond': 11, 'door': 12, 'ear': 13, 'guitar': 14, 'hamburger': 15,
    'hammer': 16, 'hand': 17, 'hat': 18, 'ladder': 19, 'leaf': 20,
    'lion': 21, 'pencil': 22, 'rabbit': 23, 'scissors': 24, 'shoe': 25,
    'star': 26, 'sword': 27, 'The_Eiffel_Tower': 28, 'tree': 29}

num_class = len(class_list)
one_hot_dict = {v: k for k, v in class_list.items()}

def get_class_name(X):
    if np.isscalar(X):
        return one_hot_dict[X]
    else:
        class_names = []
        for i in X:
            class_names.append(one_hot_dict[i])
        return class_names

def one_hot_encode(X):
    # return encoding vector 
    if np.isscalar(X):
        vector = np.zeros((num_class, 1), dtype=np.float32)
        vector[X] = 1
        return vector
    else:
        vector = np.zeros((len(X), num_class))
        for i, j in enumerate(X):
            vector[i, j] = 1
        return vector 
