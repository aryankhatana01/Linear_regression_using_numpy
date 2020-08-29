import numpy as np

X = np.array([2,4,6,8, 10],dtype=np.float32)

# Y = 2 * X + 1
Y = np.array([5,9,13,17, 21],dtype=np.float32)

w = 0.0
b = 0.0

# Forward Propagation
def forward(X):
    return (w*X) + b


# calcuate loss :-  Mean squared error
def loss(y, y_pred):
    return ((y_pred - y)**2).mean()


# gradients
# dl/dw = 1/N * 2x * (w*X + b)
def gradientW(x, y, y_pred):
    return np.dot(2*x, y_pred - y).mean()



def gradientB(x,y,y_pred):
    return np.dot(2, y_pred - y).mean()



def train(X, Y, learning_rate, epochs):
        
    for epoch in range(epochs):
        global w
        global b
        
        # Foward Pass
        y_pred = forward(X)
        
        # Loss
        l = loss(Y,y_pred)
        
        # Gradient 
        dw = gradientW(X, Y, y_pred)
        db = gradientB(X, Y, y_pred)
        
        # updating slope
        w -= (learning_rate*dw)
        b -= (learning_rate*db)
        
        if epoch%5==0:
            print('epoch', epoch+1, 'loss =', l)
        

    
epochs = 100000
learning_rate = 0.0001

train(X,Y,learning_rate,epochs)

# for epoch in range(epochs):
#     # predict = forward pass
#     y_pred = forward(X)

#     # loss
#     l = loss(Y, y_pred)
    
#     # calculate gradients
#     dw = gradient(X, Y, y_pred)

#     # update weights
#     w -= learning_rate * dw
    
    # if epoch%1==0:
    #     print('epoch', epoch+1, 'loss =', l)
print('Predicted', forward(12))