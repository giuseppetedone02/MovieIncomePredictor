from sklearn.metrics import mean_squared_error, mean_absolute_error

# Supponiamo che y_true e y_pred siano i valori veri e predetti rispettivamente
y_true = [3.0, -0.5, 2.0, 7.0]
y_pred = [2.5, 0.0, 2.0, 8.0]

# Calcolo di MSE e MAE
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)

print(f'MSE: {mse}')
print(f'MAE: {mae}')