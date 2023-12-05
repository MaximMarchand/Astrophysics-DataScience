import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def neuron_activation(network, layer):
    return network[layer]['W'] @ network[layer]['I'] 


def softmax(x, derivative=False):
    exp_shifted = np.exp(x - x.max()) # for stability the values are shifted

    if derivative:
        return exp_shifted / np.sum(exp_shifted, axis=0) * (1 - exp_shifted / np.sum(exp_shifted, axis=0))
    
    else:
        return exp_shifted / np.sum(exp_shifted, axis=0)


def transfert_sigmoid(network, layer, derivative=False):
    """
    /!\ This function should not be called directly
    """
    Y = network[layer]['Y']
    if not derivative:
        return 1 / (1 + np.exp(-Y))
    else:
        return np.exp(-Y) / (1 + np.exp(-Y))**2

def transfert_ReLU(network, layer, derivative=False):
    """
    /!\ This function should not be called directly
    """
    Y = network[layer]['Y']
    lower = Y < 0.0    

    if not derivative:
        Y[lower] = 0.0
        return Y
    
    else:
        Y[lower] = 0.0
        Y[~lower] = 1.0
        return Y

def transfert_ReLU_dying(network, layer, derivative=False):
    """
    /!\ This function should not be called directly
    """
    Y = network[layer]['Y']
    lower = Y < 0.0
    
    if not derivative:
        Y[lower] = 0.01 * Y[lower]
        return Y

    else:
        Y[lower] = 0.01
        Y[~lower] = 1.0
        return Y

def transfert(network, layer, derivative=False, mode='sig'):
    if   mode == 'sig':
        return transfert_sigmoid(network, layer, derivative)
    elif mode == 'ReLU':
        return transfert_ReLU(network, layer, derivative)
    elif mode == 'ReLU_dying': # mode == 'ReLU_dying'
        return transfert_ReLU_dying(network, layer, derivative)
    else:
        raise ValueError("{} is not a valid keyword.",format(mode))

def generate_neural_network(num_neurons_input, num_hidden_layers, num_neurons_hidden, num_neurons_output):

    output = []
    
    # Creating the input layer ----------------------------------------------------------
    input_layer = {}
    
    input_layer['I']  = np.hstack([np.zeros(num_neurons_input), np.ones(1)])
    input_layer['W']  = np.random.normal(loc=0.0, scale=1/np.sqrt(len(input_layer['I'])/2), size=(num_neurons_input, len(input_layer['I']))) 
    input_layer['Y']  = np.zeros(shape=num_neurons_input)
    input_layer['O']  = np.zeros(shape=num_neurons_input)
    input_layer['dl'] = np.zeros(shape=num_neurons_input)
    input_layer['DW'] = np.zeros(shape=input_layer['W'].shape)
    
    output.append(input_layer)
    
    # Creating the hidden layers --------------------------------------------------------
    # Looping on each hidden layer. As we have already added the input layer, output[i] 
    # corresponds to the previous layer
    for i in range(num_hidden_layers):
        ilayer = {}
        
        ilayer['I']  = np.hstack([np.zeros(len(output[i]['O'])), np.ones(1)])
        ilayer['W']  = np.random.normal(loc=0.0, scale=1/np.sqrt(len(ilayer['I'])/2), size=(num_neurons_hidden, len(ilayer['I'])))
        ilayer['Y']  = np.zeros(shape=num_neurons_hidden)
        ilayer['O']  = np.zeros(shape=num_neurons_hidden)
        ilayer['dl'] = np.zeros(shape=num_neurons_hidden)
        ilayer['DW'] = np.zeros(shape=ilayer['W'].shape)
        
        output.append(ilayer)

    # Creating the output layer ----------------------------------------------------------
    output_layer = {}

    output_layer['I']  = np.hstack([np.zeros(len(output[-1]['O'])), np.ones(1)])
    output_layer['W']  = np.random.normal(loc=0.0, scale=1/np.sqrt(len(output_layer['I'])/2), size=(num_neurons_output, len(output_layer['I'])))
    output_layer['Y']  = np.zeros(shape=num_neurons_output)
    output_layer['O']  = np.zeros(shape=num_neurons_output)
    output_layer['dl'] = np.zeros(shape=num_neurons_output)
    output_layer['DW'] = np.zeros(shape=output_layer['W'].shape)
    
    output.append(output_layer)
    
    # Checking the shapes ------------------------
    # (Only for debugging purposes)
    for i in range(len(output)):
        print("W_{0} : {1} | I_{0} : {2} | Y_{0} : {3} | O_{0} : {4} | DW_{0} : {5} | dl_{0} : {6}".format(i, output[i]['W'].shape, output[i]['I'].shape, output[i]['Y'].shape, output[i]['O'].shape, output[i]['DW'].shape, output[i]['dl'].shape))
    # --------------------------------------------
    
    return output

def forward_propagation(network, entry, transfert_mode): # Ajouter un paramètre "transfert_mode" pour indiquer le type de fonction de transfert
    # Input layer
    network[0]['I'][:-1] = entry    # Initializing first layer with data
    network[0]['Y'] = neuron_activation(network, layer=0)
    network[0]['O'] = transfert(network, layer=0, derivative=False, mode=transfert_mode)
    
    for l in range(1, len(network)-1):
        network[l]['I'] = np.hstack([network[l-1]['O'], np.ones(1)])
        network[l]['Y'] = neuron_activation(network, layer=l)
        network[l]['O'] = transfert(network, layer=l, derivative=False, mode=transfert_mode)
    
    # Output layer
    network[-1]['I'] = np.hstack([network[-2]['O'], np.ones(1)])
    network[-1]['Y'] = neuron_activation(network, layer=-1)
    network[-1]['O'] = softmax(network[-1]['Y'])

    return network


def cost(result, target):
    return np.sqrt(np.sum((result - target)**2))

def cost_gradient(result, target):
    return (result - target) / cost(result, target)


def backward_propagation(network, target, transfert_mode):
    # Looping backward from the output layer to the input layer
    for l in range(len(network)-1, -1, -1):
        if l == len(network)-1: # output layer
            gradC = cost_gradient(network[-1]['O'], target)
            derivS = softmax(network[-1]['Y'], derivative=True)
            network[-1]['dl'] = gradC * derivS

        else:
            derivatives = transfert(network, layer=l, derivative=True, mode=transfert_mode)
            M = network[l+1]['W'][:, :-1].T.copy()
            for i in range(len(derivatives)): # Vérifier si c'est bien toutes les lignes
                M[i, :] = M[i, :] * derivatives[i]
            
            network[l]['dl'][:] = (M@network[l+1]['dl'])[:]
            
    return network
        

def compute_delta_weights(network):
    for l in range(len(network)):
        d = network[l]['dl']
        I = network[l]['I']
        network[l]['DW'] = I[None, :] * d[:, None]
        
    return network

def update_network(network, data, clas, transfert_mode):
    # Setting input layer with data
    network[0]['I'] = np.hstack([data, np.ones(1)])
    network = forward_propagation(network, data, transfert_mode)
    network = backward_propagation(network, clas, transfert_mode)
    network = compute_delta_weights(network)
    
    return network

def train_network(network, num_epoch, learning_rate, training_set, transfert_mode):
    
    vec_cost = np.ndarray(num_epoch)
    
    for iepoch in tqdm(range(1, num_epoch+1)):
        for i in range(len(training_set['vectors'])):
            network = update_network(network, training_set['vectors'][i], training_set['targets'][i], transfert_mode)
            
            # Updating the weights matrices
            for l in range(len(network)):
                network[l]['W'] = network[l]['W'] - learning_rate * network[l]['DW']
            
        vec_cost[iepoch-1] = cost(network[-1]['O'], training_set['targets'][i])#/iepoch

            
    return network, vec_cost
