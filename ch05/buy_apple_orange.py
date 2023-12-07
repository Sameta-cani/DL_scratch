from layer_naive import MulLayer, AddLayer

# Initialize values
apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

# Create layers
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()

# Forward pass
apple_price = mul_apple_layer.forward(apple, apple_num)        # (1)
orange_price = mul_orange_layer.forward(orange, orange_num)    # (2)
all_price = add_apple_orange_layer.forward(apple_price, orange_price)    # (3)
total_price = mul_tax_layer.forward(all_price, tax)            # (4)

# Backward pass
dtotal_price = 1
dall_price, dtax = mul_tax_layer.backward(dtotal_price)        # (4)
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)    # (3)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)            # (2)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)            # (1)

# Print results
print(f"Total Price: {int(total_price)}")
print(f"dApple: {dapple}")
print(f"dApple_num: {int(dapple_num)}")
print(f"dOrange: {dorange}")
print(f"dOrange_num: {int(dorange_num)}")
print(f"dTax: {dtax}")