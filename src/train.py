def train_model():
    import os
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    data = pd.DataFrame({
        'x': [1,2,3,4,5],
        'y': [2.2,4.1,6.0,8.1,10.2]
    })
    X = data[['x']]
    y = data['y']
    model = LinearRegression()
    model.fit(X, y)
    # Create directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    # Save model
    import pickle
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)
    