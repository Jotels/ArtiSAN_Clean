from ase.collections import g2, s22
import numpy as np

def parse_formula(formula):
    # Parses a molecule into a dictionary of element counts
    elements = {}
    current_element = ""
    current_count = ""
    for char in formula:
        if char.isupper():
            if current_element:
                elements[current_element] = int(current_count) if current_count else 1
            current_element = char
            current_count = ""
        elif char.islower():
            current_element += char
        elif char.isdigit():
            current_count += char
    if current_element:
        elements[current_element] = int(current_count) if current_count else 1
    return elements


