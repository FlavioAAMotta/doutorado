import random

random.seed(42)

def expression_with_no_variables(expression):
    return any(char in ["x", "y", "z", "w"] for char in expression)

def get_expression(limit):
    expression = generate_expression(limit)
    while not expression_with_no_variables(expression):
        expression = generate_expression(limit)
    return expression

def generate_expression(limit):
    if limit == 1:
        return generate_factor(limit)
    else:
        choice = random.randint(1, 3)
        if choice == 1:
            return generate_term(limit)
        elif choice == 2:
            return generate_expression(limit-1) + "+" + generate_term(limit)
        else:
            return generate_expression(limit-1) + "-" + generate_term(limit)
    
def generate_term(limit):
    if limit == 1:
        return generate_factor(limit)
    else:
        choice = random.randint(1, 4)
        if choice == 1:
            return generate_factor(limit)
        elif choice == 2:
            return generate_term(limit-1) + "*" + generate_factor(limit)
        elif choice == 3:
            return generate_term(limit-1) + "/" + generate_factor(limit)
        else:
            return generate_term(limit-1) + "%" + generate_factor(limit)

def generate_factor(limit):
    choice = random.randint(1, 3)
    if choice == 1:
        return generate_number()
    elif choice == 2:
        return generate_variable()
    else:
        return "(" + generate_expression(limit-1) + ")"

def generate_number():
    return ''.join([generate_digit() for _ in range(random.randint(1, 3))])

def generate_variable():
    return random.choice(["x", "y", "z", "w"])

def generate_digit():
    return str(random.randint(0, 9))

# Exemplo de uso
for _ in range(5):
    print(get_expression(4))
