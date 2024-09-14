import random

random.seed(42)

def get_boolean_expression(limit):
    expression = generate_boolean_expression(limit)
    return expression

def generate_boolean_expression(limit):
    if limit <= 1:
        return generate_boolean_term(limit)
    else:
        choice = random.randint(1, 3)
        if choice == 1:
            return generate_boolean_term(limit)
        elif choice == 2:
            return "(" + generate_boolean_expression(limit-1) + ") and (" + generate_boolean_term(limit) + ")"
        else:
            return "(" + generate_boolean_expression(limit-1) + ") or (" + generate_boolean_term(limit) + ")"

def generate_boolean_term(limit):
    if limit <= 1:
        return generate_boolean_factor(limit)
    else:
        choice = random.randint(1, 2)
        if choice == 1:
            return generate_boolean_factor(limit)
        else:
            return "not (" + generate_boolean_term(limit-1) + ")"

def generate_boolean_factor(limit):
    if limit <= 1:
        return generate_comparison_or_boolean()
    else:
        choice = random.randint(1, 2)
        if choice == 1:
            return generate_comparison_or_boolean()
        else:
            return "(" + generate_boolean_expression(limit-1) + ")"

def generate_comparison_or_boolean():
    choice = random.randint(1, 2)
    if choice == 1:
        return generate_comparison()
    else:
        return generate_boolean_value()

def generate_comparison():
    left = generate_arithmetic_expression(1)
    operator = random.choice(["==", "!=", "<", ">", "<=", ">="])
    right = generate_arithmetic_expression(1)
    return "(" + left + " " + operator + " " + right + ")"

def generate_arithmetic_expression(limit):
    if limit <= 1:
        return generate_number_or_variable()
    else:
        choice = random.randint(1, 2)
        if choice == 1:
            return generate_number_or_variable()
        else:
            return "(" + generate_arithmetic_expression(limit-1) + random.choice(["+", "-", "*", "/"]) + generate_arithmetic_expression(limit-1) + ")"

def generate_number_or_variable():
    choice = random.randint(1, 2)
    if choice == 1:
        return generate_number()
    else:
        return generate_variable()

def generate_variable():
    return random.choice(["x", "y", "z", "w"])

def generate_number():
    # Decide se vai gerar '0' ou um número sem zeros à esquerda
    if random.choice([True, False]):
        return '0'
    else:
        num_digits = random.randint(1, 2)
        first_digit = generate_non_zero_digit()
        if num_digits == 1:
            return first_digit
        else:
            other_digits = ''.join([generate_digit() for _ in range(num_digits -1)])
            return first_digit + other_digits

def generate_non_zero_digit():
    return str(random.randint(1, 9))

def generate_digit():
    return str(random.randint(0, 9))

def generate_boolean_value():
    return random.choice(["True", "False"])

# Exemplo de uso
variables = {'x': 5, 'y': 10, 'z': 3, 'w': 7}

for _ in range(5):
    expr = get_boolean_expression(4)
    try:
        # Avaliar a expressão com os valores de variáveis fornecidos
        result = eval(expr, {}, variables)
        # Converter o resultado booleano para inteiro (True -> 1, False -> 0)
        result_int = int(bool(result))
        print(f"Expressão: {expr}, Resultado: {result_int}")
    except Exception as e:
        print(f"Expressão: {expr}, Erro: {e}")
