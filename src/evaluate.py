from utilis import Functional

def evaluate(model, x_test, y_test):
    y_pred = [model(x) for x in x_test]
    y_soft = Functional().softmax(y_pred)
    prediction = []
    for ys in y_soft:
        y_temp = [y.data for y in ys]
        prediction.append(max(range(len(y_temp)), key=y_temp.__getitem__))

    actuals = [model.class_to_index[y] for y in y_test]
    accuracy = sum(p == a for p, a in zip(prediction, actuals)) / len(actuals) * 100
    print(f"Accuracy: {accuracy:.4f}")
