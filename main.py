def read_float(message: str) -> float:
    while True:
        try:
            return float(input(message))
        except ValueError:
            print("Incorrect input!")


def proceed(question: str) -> bool:
    while True:
        f = input(question + " (yes/No)? ").lower()
        if f == "yes":
            print()
            return True
        elif f == "" or f == "no":
            return False
        else:
            print("Expected answer 'yes' or 'no'")