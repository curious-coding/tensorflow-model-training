import os

from dataclasses import dataclass


@dataclass
class Images:
    directory: str = os.path.dirname(os.path.realpath(__file__))
    training: str = directory + "/animals-or-plants"
    validation: str = directory + "/validation-animals-or-plants"
    animals: str = training + "/animals"
    plants: str = training + "/plants"
    valid_animals: str = validation + "/animals"
    valid_plants: str = validation + "/plants"
