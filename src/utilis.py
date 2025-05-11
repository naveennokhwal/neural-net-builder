from value import Value
import random

class Functional:
    def csv_to_lists(self, features_df, labels_df):
        features = []
        labels = []

        try:
            features = features_df.values.tolist()  # Convert DataFrame to list of lists
        except Exception as e:
            raise ValueError(f"Error converting features CSV to list: {e}")
        # Handle labels data
        try:
            if labels_df.shape[1] != 1:
                raise ValueError("Label CSV file must have only one column.")
            labels = labels_df.values.flatten().tolist()  # Convert DataFrame to 1D list
        except Exception as e:
            raise ValueError(f"Error converting labels CSV to list: {e}")
        if not features:
            print("Warning: Feature file is empty.")
        if not labels:
            print("Warning: Label file is empty.")
        if len(features) != len(labels):
            raise ValueError(f"Number of features ({len(features)}) does not match number of labels ({len(labels)}).")
        return features, labels

    def one_hot_encode_labels(self, labels_list):
        # Get unique classes
        classes = list(set(labels_list))
        one_hot_encoded_list = []

        # Create a mapping from class to index, or class to vector
        class_to_index = {cls: i for i, cls in enumerate(classes)}

        # Iterate through the labels and set the corresponding one-hot encoded row
        for label in labels_list:
            class_index = class_to_index[label]
            vector = [0] * len(classes)
            vector[class_index] = 1
            one_hot_encoded_list.append(vector)

        return one_hot_encoded_list, class_to_index
    
    def mse(self, y_pred, y_actual):
        mse = sum(((yout - ygt)**2 for yout, ygt in zip(y_pred, y_actual)), Value(0.0))
        return mse

    def softmax(self, y_pred):
        y_soft = []
        for y in y_pred:
            den = sum((v.exp() for v in y), Value(0.0))
            y_soft.append([v.exp()/den for v in y])
        return y_soft
    
    def cross_entropy_loss(self, y_pred, y_actual):
        encoded_labels, class_to_index = self.one_hot_encode_labels(y_actual)
        total_loss = Value(0.0)
        for youts, ygts in zip(y_pred, encoded_labels):
           ce = sum((yout.log() * ygt for yout,ygt in zip(youts, ygts)), Value(0.0))
           total_loss+= ce
        
        return -1*total_loss/len(y_actual), class_to_index
    
    def train_test_split(self, features, labels, train_size, shuffle = True):
        features_copy = features[:]
        labels_copy = labels[:]
        if len(features_copy) != len(labels_copy):
            raise ValueError(f"Number of features ({len(features_copy)}) does not match number of labels ({len(labels_copy)}).")

        if shuffle == True:
                combined = list(zip(features_copy, labels_copy))
                random.shuffle(combined)
                features_copy, labels_copy = zip(*combined)

        train_len = int(len(labels_copy)*train_size)
        x_train = features_copy[:train_len]
        x_test = features_copy[train_len:]

        y_train = labels_copy[:train_len]
        y_test = labels_copy[train_len:]

        return x_train, x_test, y_train, y_test

