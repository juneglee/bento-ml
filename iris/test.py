import bentoml

loaded_model = bentoml.sklearn.load_model("iris_clf:latest")

result = loaded_model.predict([[5.9, 3. , 5.1, 1.8]])  # => array(2)

print(result)