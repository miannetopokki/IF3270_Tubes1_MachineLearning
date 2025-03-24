from lib.value import Value
# loss = sum([sum((yout_i - ygt_i)**2 for ygt_i, yout_i in zip(ygt, [yout] if isinstance(yout, Value) else yout)) for ygt, yout in zip(y_val, val_pred)]) / N_val
def mean_squared_error(y_true, y_pred):
    N = sum(len(y) for y in y_true)  
    total_loss = 0  
    for ygt, yout in zip(y_true, y_pred):
        # yout berbentuk list jika hanya satu neuron
        yout = [yout] if isinstance(yout, Value) else yout
        
        # error per elemen
        total_loss += sum((yout_i - ygt_i) ** 2 for ygt_i, yout_i in zip(ygt, yout))

    return total_loss / N
