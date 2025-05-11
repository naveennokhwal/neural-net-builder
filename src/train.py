from model import MLP
from utilis import Functional
import pandas as pd
from evaluate import evaluate


#step 1: do required data preprocessing
## output should be two different files( feature and label) with data type of list
## feature : 2-D list, label: 1-D list

# step 1.1: loading
path = r"C:\Learning\Machine-Learning\Deep_Learning_WorkSpace\projects\neural-network-project\data\processed\Iris_noindex.csv"
df = pd.read_csv(path)

feature = df[['SepalLengthCm',	'SepalWidthCm',	'PetalLengthCm',	'PetalWidthCm']]
label = df[['Species']]

# step 1.2: convert into list data type
feature_list, label_list = Functional().csv_to_lists(feature, label)

#step 1.3: train-test split
X_train, X_test, y_train, y_test = Functional().train_test_split(feature_list, label_list, train_size=0.7, shuffle = True)


# step 2: Model training
# step 2.1: make an object of model class
net = MLP(4,3)

# step 2.2: set hyper-parameters
epochs = 200
lr = 0.1
class_to_index = None

# step 2.3: training loop
for epoch in range(epochs):
    #step 2.3.1: make predictions
    y_pred = [net(x) for x in X_train]

    y_soft = Functional().softmax(y_pred)

    #step 2.3.2: calculate cross-entropy loss
    loss, class_to_index = Functional().cross_entropy_loss(y_soft, y_train)

    #step 2.3.3: set gradients of all the parameters to zero
    for p in net.parameters():
        p.grad = 0.0 

    #step 2.3.4: calcualte gradients of all the parameters using backpropogation
    loss.backward()
    
    #step 2.3.5: update parameters to minimize the loss 
    for p in net.parameters():
        p.data += -lr*p.grad

    #step 2.3.6: print statement for each epoch
    print(f"epoch: {epoch+1}/{epochs} | Loss: {loss}")

# to save computational graph of the model
# draw_graph(loss).render("computational graph")
#step 3: save the model
net.save_model(other=class_to_index)
net.class_to_index = class_to_index

evaluate(net, X_test, y_test)
