from layer_naive import MulLayer

# Initialize values
apple = 100
apple_num = 2
tax = 1.1

# Create multiplication layers
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# Forward pass
apple_price = mul_apple_layer.forward(apple, apple_num)
total_price = mul_tax_layer.forward(apple_price, tax)

# Backward pass
dtotal_price = 1
dapple_price, dtax = mul_tax_layer.backward(dtotal_price)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

# Print results
print(f"Total Price: {int(total_price)}")
print(f"dApple: {dapple}")
print(f"dApple_num: {int(dapple_num)}")
print(f"dTax: {dtax}")